import cv2
import numpy as np
import time
import os
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor

# Import specific functions from the refactored modules to avoid circular imports
from contours import find_all_card_contours_from_binary
from transforms import four_point_transform, extract_top_left_corner
from symbols import extract_symbols_from_corner, enhanced_rank_classification
from suit_classifier import classify_suit_v7

class LiveCardDetector:
    def __init__(self, rank_templates, suit_templates, suit_color_prototypes, camera_source=None,
                 redetect_interval=6, max_workers=3, min_card_area=25000):
        """
        Optimized LiveCardDetector with:
         - Optical-flow tracking of card corner pts between full detections
         - Async classification (thread pool) for tracked frames
         - Periodic full redetection to recover from drift/occlusion

        Parameters added (configurable):
         - redetect_interval: run full contour detection every N frames (default 6)
         - max_workers: threadpool workers for async classification
         - min_card_area: minimum area for considering contours (default 25000)
        """
        self.rank_templates = rank_templates
        self.suit_templates = suit_templates
        self.suit_color_prototypes = suit_color_prototypes

        if camera_source is None:
            camera_source = 0
        print(f"Intentando conectar a cámara: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara: {camera_source}\nVerifica DroidCam / URL.")
        # Keep reasonably high resolution but you can lower it externally if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Detection / tracking parameters
        self.min_card_area = min_card_area
        self.redetect_interval = redetect_interval

        # Threading for classification tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}  # key -> future

        # State
        self.paused = False
        self.last_frame = None
        self.last_annotated = None

        # Historial por “celda” (posición) para estabilizar cada carta
        self.per_card_histories = {}
        self.history_len = 6
        self.stable_threshold = 3  # mínimo apariciones para considerar estable

        # Guardar último warp por key para captura individual si quieres
        self.last_warps = {}

        # Optical-flow tracking state
        # key -> 4x2 float32 array of corner points
        self.per_card_prev_pts = {}
        # key -> gray image where prev_pts were sampled
        self.per_card_last_gray = {}

        # Last synchronous/async classification result per key
        # key -> dict { 'rank', 'suit', 'scores':(rank_score,suit_score), 'debug': {...}, 'timestamp': t }
        self.per_card_last_result = {}

        self.detection_cooldown = 0.8
        self.last_announce_time = 0
        self.last_announced_cards = set()  # {(rank,suit,key)}

    def _cell_key(self, contour):
        """
        Genera una clave de celda para agrupar detecciones de la misma carta
        basado en el centro aproximado del contorno.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Cuantizar para evitar variaciones pequeñas
        return (cx // 40, cy // 40)

    def _update_history(self, key, rank, suit):
        if key is None:
            return
        dq = self.per_card_histories.get(key)
        if dq is None:
            dq = deque(maxlen=self.history_len)
            self.per_card_histories[key] = dq
        dq.append((rank, suit))

    def _stable_vote(self, key):
        dq = self.per_card_histories.get(key)
        if dq is None or len(dq) < self.stable_threshold:
            return None, None
        ranks = [r for r,_ in dq]
        suits = [s for _,s in dq]
        r_cnt = Counter(ranks).most_common(1)[0]
        s_cnt = Counter(suits).most_common(1)[0]
        if r_cnt[1] >= self.stable_threshold and s_cnt[1] >= self.stable_threshold:
            return r_cnt[0], s_cnt[0]
        return None, None

    def _classify_task(self, rank_sym, suit_sym, corner_rgb, key):
        """
        Synchronous classification task (can be run in a thread).
        Returns a dict with results.
        """
        rank_match, rank_score = enhanced_rank_classification(rank_sym, self.rank_templates)
        suit_match, suit_score, suit_debug = classify_suit_v7(
            suit_sym, corner_rgb, self.suit_templates, self.suit_color_prototypes
        )
        return {
            "rank": rank_match,
            "suit": suit_match,
            "scores": (rank_score, suit_score),
            "debug": suit_debug
        }

    def _classification_callback(self, key, fut):
        """
        Callback after async classification completes.
        Updates histories and last results.
        """
        try:
            res = fut.result()
        except Exception as e:
            # classification failed in worker
            print(f"[Async classify] Error para key {key}: {e}")
            return

        rank_match = res["rank"]
        suit_match = res["suit"]
        rank_score, suit_score = res["scores"]
        debug = res.get("debug", {})

        # Update history & last_result
        self._update_history(key, rank_match, suit_match)
        self.per_card_last_result[key] = {
            "rank": rank_match,
            "suit": suit_match,
            "scores": (rank_score, suit_score),
            "debug": debug,
            "timestamp": time.time()
        }

        # Announce if stable now
        stable_rank, stable_suit = self._stable_vote(key)
        if stable_rank is not None and stable_suit is not None:
            card_id = (stable_rank, stable_suit, key)
            now = time.time()
            if (now - self.last_announce_time > self.detection_cooldown and
                card_id not in self.last_announced_cards):
                print(f"✓ Carta estable (async): {stable_rank} de {stable_suit} (scores R={rank_score:.2f}, S={suit_score:.2f})")
                self.last_announced_cards.add(card_id)
                self.last_announce_time = now

    def process_frame_multi(self, frame, frame_count):
        """
        Procesa el frame y detecta múltiples cartas.
        Mejora: combina redetección periódica con tracking óptico entre detecciones.
        Devuelve:
          frame_annotated,
          detecciones = [ { 'key':key, 'rank':r, 'suit':s, 'stable':bool,
                            'scores':(rank_score,suit_score), 'contour_pts':pts } ]
        """
        annotated = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(cur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        detections = []

        # Decide whether to run full redetection this frame
        do_redetect = (frame_count % self.redetect_interval == 0) or (len(self.per_card_prev_pts) == 0)

        if do_redetect:
            # Full detection: find all card contours and re-initialize trackers for them
            candidates = find_all_card_contours_from_binary(binary, min_area=self.min_card_area)

            for approx, area in candidates:
                key = self._cell_key(approx)
                # store the 4 corner points for tracking; approximate may be Nx1x2 shape
                src_pts = approx.reshape(-1, 2).astype(np.float32)
                # Ensure we have 4 points; if approx returns 4 points already ok; otherwise try to pick 4 extremes
                if src_pts.shape[0] != 4:
                    # fallback: compute bounding rect corners
                    x,y,wc,hc = cv2.boundingRect(approx)
                    src_pts = np.array([[x,y],[x+wc,y],[x+wc,y+hc],[x,y+hc]], dtype=np.float32)

                # Save for tracking
                self.per_card_prev_pts[key] = src_pts.copy()
                self.per_card_last_gray[key] = cur_gray.copy()

                # Warp & extract symbols synchronously for a newly detected card (immediate result)
                try:
                    warped = four_point_transform(frame_rgb, src_pts.reshape(4,1,2), width=300, height=420)
                    corner = extract_top_left_corner(warped)
                    _, symbols = extract_symbols_from_corner(corner)
                except Exception:
                    symbols = []

                if len(symbols) < 2:
                    # incomplete symbols -> draw in yellow
                    cv2.drawContours(annotated, [approx], -1, (0, 255, 255), 2)
                    continue

                rank_sym = symbols[0]
                suit_sym = symbols[1]

                # Do synchronous classification for a freshly detected card to get immediate feedback
                res = self._classify_task(rank_sym, suit_sym, corner, key)
                rank_match = res["rank"]
                suit_match = res["suit"]
                rank_score, suit_score = res["scores"]
                debug = res.get("debug", {})

                # update history/result
                self._update_history(key, rank_match, suit_match)
                self.per_card_last_result[key] = {
                    "rank": rank_match,
                    "suit": suit_match,
                    "scores": (rank_score, suit_score),
                    "debug": debug,
                    "timestamp": time.time()
                }

                # draw as detected (green if stable quickly, else cyan)
                stable_rank, stable_suit = self._stable_vote(key)
                stable = (stable_rank is not None and stable_suit is not None)
                color = (0, 255, 0) if stable else (255, 255, 0)
                cv2.drawContours(annotated, [approx], -1, color, 3)

                # store warp
                self.last_warps[key] = warped

                detections.append({
                    "key": key,
                    "rank": rank_match,
                    "suit": suit_match,
                    "stable": stable,
                    "scores": (rank_score, suit_score),
                    "contour_pts": src_pts
                })

        else:
            # Use optical flow to track previously detected cards
            # For each tracked card, try to calcOpticalFlowPyrLK from last_gray -> cur_gray
            keys_to_remove = []
            for key, prev_pts in list(self.per_card_prev_pts.items()):
                prev_gray = self.per_card_last_gray.get(key)
                if prev_gray is None or prev_pts is None:
                    keys_to_remove.append(key)
                    continue

                # Prepare points in shape (N,1,2)
                p0 = prev_pts.reshape(-1, 1, 2).astype(np.float32)
                try:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                                           winSize=(21,21), maxLevel=3,
                                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                except Exception:
                    # Optical flow can fail in some cases; schedule re-detection
                    keys_to_remove.append(key)
                    continue

                if p1 is None or st is None:
                    keys_to_remove.append(key)
                    continue

                good = st.reshape(-1) == 1
                if np.count_nonzero(good) < 4:
                    # tracking failed for enough points -> re-detect later
                    keys_to_remove.append(key)
                    continue

                tracked_pts = p1.reshape(-1, 2)
                # update tracker state
                self.per_card_prev_pts[key] = tracked_pts.copy()
                self.per_card_last_gray[key] = cur_gray.copy()

                # Warp using tracked_pts and extract symbols
                try:
                    warped = four_point_transform(frame_rgb, tracked_pts.reshape(4,1,2), width=300, height=420)
                    corner = extract_top_left_corner(warped)
                    _, symbols = extract_symbols_from_corner(corner)
                except Exception:
                    symbols = []

                if len(symbols) < 2:
                    # draw contour from tracked pts
                    int_pts = tracked_pts.reshape(-1,2).astype(np.int32)
                    cv2.polylines(annotated, [int_pts], True, (0,255,255), 2)
                    continue

                rank_sym = symbols[0]
                suit_sym = symbols[1]

                # If there is already a running future for this key, skip submitting another
                if key in self.futures and not self.futures[key].done():
                    # Use last known result for drawing
                    last = self.per_card_last_result.get(key)
                    if last:
                        stable_rank, stable_suit = self._stable_vote(key)
                        stable = (stable_rank is not None and stable_suit is not None)
                        color = (0, 255, 0) if stable else (255, 255, 0)
                        int_pts = tracked_pts.reshape(-1,2).astype(np.int32)
                        cv2.polylines(annotated, [int_pts], True, color, 3)
                        label = f"{last['rank']} {last['suit']}"
                        x,y,wc,hc = cv2.boundingRect(int_pts)
                        cv2.rectangle(annotated, (x, y-25), (x+wc, y), (0,0,0), -1)
                        cv2.putText(annotated, label, (x+5, y-7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if stable else (0,255,255), 1)
                    continue

                # Submit async classification to threadpool (non-blocking)
                future = self.executor.submit(self._classify_task, rank_sym, suit_sym, corner, key)
                # store future and bind callback
                self.futures[key] = future
                future.add_done_callback(lambda fut, k=key: self._classification_callback(k, fut))

                # draw placeholder (will be updated when async done)
                int_pts = tracked_pts.reshape(-1,2).astype(np.int32)
                last = self.per_card_last_result.get(key)
                if last:
                    stable_rank, stable_suit = self._stable_vote(key)
                    stable = (stable_rank is not None and stable_suit is not None)
                    color = (0, 255, 0) if stable else (255, 255, 0)
                    label = f"{last['rank']} {last['suit']}"
                else:
                    color = (255, 255, 0)
                    label = "..."
                cv2.polylines(annotated, [int_pts], True, color, 3)
                x,y,wc,hc = cv2.boundingRect(int_pts)
                cv2.rectangle(annotated, (x, y-25), (x+wc, y), (0,0,0), -1)
                cv2.putText(annotated, label, (x+5, y-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if last else (0,255,255), 1)

                # store warp
                self.last_warps[key] = warped

                detections.append({
                    "key": key,
                    "rank": self.per_card_last_result.get(key, {}).get("rank", "Unknown"),
                    "suit": self.per_card_last_result.get(key, {}).get("suit", "Unknown"),
                    "stable": False,
                    "scores": self.per_card_last_result.get(key, {}).get("scores", (0.0,0.0)),
                    "contour_pts": tracked_pts
                })

            # remove trackers that failed
            for k in keys_to_remove:
                if k in self.per_card_prev_pts:
                    del self.per_card_prev_pts[k]
                if k in self.per_card_last_gray:
                    del self.per_card_last_gray[k]
                # keep last_result and history to allow showing last known label for some time

        # Final: draw info panel and return
        self.last_annotated = annotated.copy()
        return annotated, detections

    def draw_info_panel_multi(self, frame, detections, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Cartas detectadas: {len(detections)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Panel lateral (opcional)
        panel_w = 260
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.25, frame, 0.75, 0, frame)

        y0 = 80
        cv2.putText(frame, "DETALLES:", (w - panel_w + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        for i, det in enumerate(detections[:12]):  # limitar listado
            txt = f"{i+1}. {det['rank']} {det['suit']} {'(S)' if det['stable'] else ''}"
            cv2.putText(frame, txt, (w - panel_w + 10, y0 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0) if det["stable"] else (0,255,255), 1)

        # Instrucciones abajo
        instructions = [
            "Q: Salir | C: Capturar todas | R: Reiniciar historial",
            "ESPACIO: Pausa | A/Z: Ajustar área mínima"
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (10, h - 40 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    def run(self):
        print("\n=== MODO MULTI-CARTA EN VIVO (Optimizado) ===")
        print("Controles: Q=Salir  C=Capturar  R=Reset  ESPACIO=Pausa  A/Z=Area +/-")
        print(f"Área mínima inicial: {self.min_card_area}")
        print(f"Redetección completa cada {self.redetect_interval} frames")
        frame_count = 0
        fps_time = time.time()
        fps = 0.0

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame inválido, revisa la cámara")
                    time.sleep(0.5)
                    continue
                self.last_frame = frame.copy()
                frame_count += 1

                annotated, detections = self.process_frame_multi(frame, frame_count)

                # Compute FPS (simple moving)
                if frame_count % 15 == 0:
                    now = time.time()
                    fps = 15.0 / (now - fps_time + 1e-6)
                    fps_time = now

                # Announce newly stable cards from per_card_last_result (in case async finished)
                now = time.time()
                for key, last in list(self.per_card_last_result.items()):
                    stable_rank, stable_suit = self._stable_vote(key)
                    if stable_rank is None or stable_suit is None:
                        continue
                    card_id = (stable_rank, stable_suit, key)
                    if (now - self.last_announce_time > self.detection_cooldown and
                        card_id not in self.last_announced_cards):
                        print(f"✓ Carta estable: {stable_rank} de {stable_suit} (scores R={last['scores'][0]:.2f}, S={last['scores'][1]:.2f})")
                        self.last_announced_cards.add(card_id)
                        self.last_announce_time = now

                self.draw_info_panel_multi(annotated, detections, fps)
                display = annotated
            else:
                display = self.last_frame.copy() if self.last_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(display, "PAUSADO - ESPACIO para continuar",
                            (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("Deteccion Multi-Carta", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("Saliendo...")
                break
            elif key in (ord(' '),):
                self.paused = not self.paused
                print("Pausado" if self.paused else "Reanudado")
            elif key in (ord('r'), ord('R')):
                self.per_card_histories.clear()
                self.last_announced_cards.clear()
                self.per_card_prev_pts.clear()
                self.per_card_last_gray.clear()
                self.per_card_last_result.clear()
                print("Historial reiniciado.")
            elif key in (ord('a'), ord('A')):
                self.min_card_area += 5000
                print(f"Área mínima ahora: {self.min_card_area}")
            elif key in (ord('z'), ord('Z')):
                self.min_card_area = max(8000, self.min_card_area - 5000)
                print(f"Área mínima ahora: {self.min_card_area}")
            elif key in (ord('c'), ord('C')):
                # Capturar warps de cartas estables
                stable_warps = [ (k, self.last_warps[k]) for k in self.last_warps if self._stable_vote(k)[0] ]
                if not stable_warps:
                    print("No hay cartas estables para capturar.")
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs("captures", exist_ok=True)
                    for idx,(k,warp) in enumerate(stable_warps, start=1):
                        rank, suit = self._stable_vote(k)
                        fname = f"captures/card_{rank}_{suit}_{ts}_{idx}.jpg"
                        cv2.imwrite(fname, cv2.cvtColor(warp, cv2.COLOR_RGB2BGR))
                        print(f"Capturada: {fname}")

        self.cap.release()
        cv2.destroyAllWindows()
        # shutdown executor
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        print("Fin detección multi-carta.")