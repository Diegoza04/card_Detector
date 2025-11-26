
import cv2
import numpy as np
import math
from contours import find_contours
from features import compute_shape_metrics

def extract_symbols_from_corner(corner_rgb, min_area=50, horizontal_gap=20):
    """
    Extrae símbolos de la esquina con mejor manejo de tamaños variables y múltiples umbrales
    """
    gray = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    adaptive_min_area = max(min_area, (h * w) // 200)
    
    thresholds = []
    
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresholds.append(thresh1)
    
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    thresholds.append(thresh2)
    
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    thresholds.append(thresh3)
    
    combined_thresh = np.zeros_like(gray)
    for t in thresholds:
        combined_thresh = cv2.bitwise_or(combined_thresh, t)
    
    combined_thresh = cv2.medianBlur(combined_thresh, 3)
    
    kernel_small = np.ones((2,2), np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_small)
    
    contours_list, _ = find_contours(combined_thresh)
    boxes = []
    
    for c in contours_list:
        area = cv2.contourArea(c)
        if area < adaptive_min_area:
            continue
            
        x, y, w_box, h_box = cv2.boundingRect(c)
        
        if w_box < 5 or h_box < 5:
            continue
        if w_box / h_box > 5 or h_box / w_box > 8:
            continue
            
        boxes.append([x, y, x+w_box, y+h_box])
    
    if not boxes:
        return combined_thresh, []
    
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if not merged:
            merged.append([x1, y1, x2, y2])
        else:
            mx1, my1, mx2, my2 = merged[-1]
            if abs(y1 - my1) < 25 and (x1 - mx2) < horizontal_gap:
                merged[-1] = [min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2)]
            else:
                merged.append([x1, y1, x2, y2])
    
    merged.sort(key=lambda b: (b[1], b[0]))
    
    symbols = []
    for (x1, y1, x2, y2) in merged:
        pad = 3
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)
        
        crop = combined_thresh[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            crop_clean = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel_small)
            symbols.append(crop_clean)
    
    print(f"Símbolos extraídos: {len(symbols)} de esquina {w}x{h}")
    
    return combined_thresh, symbols

def multi_template_scores(symbol_img, templates_list):
    """
    Mejorado con más robustez para condiciones de cámara en vivo
    """
    if symbol_img is None or len(templates_list) == 0:
        return 0.0, {}
    
    H0, W0 = 32, 32
    
    symbol_norm = cv2.resize(symbol_img, (W0, W0))
    symbol_norm = cv2.equalizeHist(symbol_norm)
    
    symbol_edges = cv2.Canny(symbol_norm, 30, 150)
    symbol_inv = cv2.bitwise_not(symbol_norm)
    dist_symbol = cv2.distanceTransform(symbol_inv, cv2.DIST_L2, 3)
    
    vec_symbol = symbol_norm.flatten().astype(np.float32)
    vec_symbol /= (np.linalg.norm(vec_symbol) + 1e-6)
    
    best = 0.0
    best_detail = None
    
    for tmpl in templates_list:
        scales = [0.9, 1.0, 1.1]
        scale_scores = []
        
        for scale in scales:
            h_tmpl, w_tmpl = tmpl.shape
            new_h, new_w = int(h_tmpl * scale), int(w_tmpl * scale)
            if new_h <= 0 or new_w <= 0 or new_h > W0*2 or new_w > W0*2:
                continue
                
            tmpl_scaled = cv2.resize(tmpl, (new_w, new_h))
            tmpl_norm = cv2.resize(tmpl_scaled, (W0, W0))
            tmpl_norm = cv2.equalizeHist(tmpl_norm)
            
            res = cv2.matchTemplate(symbol_norm, tmpl_norm, cv2.TM_CCOEFF_NORMED)
            _, corr_score, _, _ = cv2.minMaxLoc(res)
            
            tmpl_edges = cv2.Canny(tmpl_norm, 30, 150)
            res_e = cv2.matchTemplate(symbol_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
            _, edge_score, _, _ = cv2.minMaxLoc(res_e)
            
            tmpl_inv = cv2.bitwise_not(tmpl_norm)
            dist_tmpl = cv2.distanceTransform(tmpl_inv, cv2.DIST_L2, 3)
            
            sym_pts = np.where(symbol_edges > 0)
            tmpl_pts = np.where(tmpl_edges > 0)
            
            ch1 = dist_tmpl[sym_pts].mean() if len(sym_pts[0]) > 0 else 50.0
            ch2 = dist_symbol[tmpl_pts].mean() if len(tmpl_pts[0]) > 0 else 50.0
            chamfer_score = np.exp(-0.5 * (ch1 + ch2) / 10.0)
            
            vec_tmpl = tmpl_norm.flatten().astype(np.float32)
            vec_tmpl /= (np.linalg.norm(vec_tmpl) + 1e-6)
            cosine = float(np.dot(vec_symbol, vec_tmpl))
            if cosine < 0:
                cosine = 0.0
            
            mean_s = np.mean(symbol_norm)
            mean_t = np.mean(tmpl_norm)
            std_s = np.std(symbol_norm)
            std_t = np.std(tmpl_norm)
            
            cov = np.mean((symbol_norm - mean_s) * (tmpl_norm - mean_t))
            ssim_score = (2 * mean_s * mean_t + 1e-6) / (mean_s**2 + mean_t**2 + 1e-6) * \
                        (2 * cov + 1e-6) / (std_s**2 + std_t**2 + 1e-6)
            ssim_score = max(0, min(1, ssim_score))
            
            combined = (0.30 * corr_score + 
                       0.20 * edge_score + 
                       0.20 * chamfer_score + 
                       0.15 * cosine +
                       0.15 * ssim_score)
            
            scale_scores.append({
                "corr": corr_score,
                "edge": edge_score,
                "chamfer": chamfer_score,
                "cosine": cosine,
                "ssim": ssim_score,
                "combined": combined,
                "scale": scale
            })
        
        if scale_scores:
            best_scale = max(scale_scores, key=lambda x: x["combined"])
            
            if best_scale["combined"] > best:
                best = best_scale["combined"]
                best_detail = best_scale
    
    return best, (best_detail if best_detail else {})

def enhanced_match_symbol_v2(symbol_img, templates_dict, symbol_type="rank"):
    if symbol_img is None or len(templates_dict) == 0:
        return "Unknown", -1.0, {}
    best_name = "Unknown"
    best_score = -1.0
    best_detail = {}
    for name, tmpl_list in templates_dict.items():
        if not isinstance(tmpl_list, list):
            tmpl_list = [tmpl_list]
        score, detail = multi_template_scores(symbol_img, tmpl_list)
        if score > best_score:
            best_score = score
            best_name = name
            best_detail = detail
    return best_name, float(best_score), best_detail

def enhanced_rank_classification(rank_symbol, rank_templates):
    """
    Clasificación mejorada de ranks con detección específica para números problemáticos.
    Mejoras para 2,3,8,10:
      - heurísticas por proyecciones y componentes conectados
      - validación extra con versiones modificadas (erosión/dilatación/rotaciones)
      - uso de métricas de forma (compute_shape_metrics) como soporte
    """
    h, w = rank_symbol.shape
    rank_symbol_enhanced = cv2.equalizeHist(rank_symbol)
    rank_symbol_denoised = cv2.bilateralFilter(rank_symbol_enhanced, 5, 50, 50)
    
    name, score, detail = enhanced_match_symbol_v2(rank_symbol_denoised, rank_templates, "rank")
    
    # Problemas conocidos
    problematic = {'8','5','6','3','10','2'}
    
    # Quick shape metrics to use in heuristics
    shape_metrics = compute_shape_metrics(rank_symbol_denoised)
    
    # If high confidence, return immediately
    if score >= 0.80:
        return name, score
    
    # Extended validation for problem digits (2,3,8,10)
    if (name in problematic) or (score < 0.65):
        # Prepare variants
        alternative_versions = []
        alternative_versions.append(('original_enhanced', rank_symbol_denoised))
        _, otsu = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alternative_versions.append(('otsu', otsu))
        adaptive_gauss = cv2.adaptiveThreshold(
            rank_symbol_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        alternative_versions.append(('adaptive_gauss', adaptive_gauss))
        kernel_thick = np.ones((2, 2), np.uint8)
        thickened = cv2.erode(rank_symbol_denoised, kernel_thick, iterations=1)
        alternative_versions.append(('thickened', thickened))
        thinned = cv2.dilate(rank_symbol_denoised, kernel_thick, iterations=1)
        alternative_versions.append(('thinned', thinned))
        # rotated variants (small angles) sometimes help for slanted camera
        for angle in (-12, -6, 6, 12):
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rot = cv2.warpAffine(rank_symbol_enhanced, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            alternative_versions.append((f'rot{angle}', rot))
        
        # Evaluate alternatives with more scales and prioritize candidate scores for specific digits
        best_candidates = {}
        scales_to_try = [0.80, 0.90, 1.0, 1.05, 1.10, 1.20]
        for version_name, version_img in alternative_versions:
            for scale in scales_to_try:
                nh, nw = int(h * scale), int(w * scale)
                if nh <= 0 or nw <= 0:
                    continue
                scaled = cv2.resize(version_img, (nw, nh))
                test_name, test_score, _ = enhanced_match_symbol_v2(scaled, rank_templates, "rank")
                if test_name not in best_candidates or test_score > best_candidates[test_name]['score']:
                    best_candidates[test_name] = {
                        'score': test_score,
                        'version': version_name,
                        'scale': scale
                    }
        
        if best_candidates:
            sorted_candidates = sorted(best_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
            # If top candidate has significant lead, accept it
            top_name, top_info = sorted_candidates[0]
            second_info = sorted_candidates[1][1] if len(sorted_candidates) > 1 else {'score': 0.0}
            score_diff = top_info['score'] - second_info['score']
            
            # Heurística específica para 8 vs 5/6/3
            if top_name == '8' or name == '8':
                # Count inner holes (8 tiene típicamente 2)
                _, binary = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                inverted = cv2.bitwise_not(binary)
                num_labels, labels = cv2.connectedComponents(inverted)
                hole_count = max(0, num_labels - 1)
                if hole_count >= 2:
                    return '8', max(top_info['score'], 0.75)
                # otherwise fallthrough to candidate decision if diff large
                if score_diff > 0.15:
                    return top_name, top_info['score']
            
            # Heurística específica para 10 (dos componentes horizontales)
            if top_name == '10' or name == '10':
                _, binary_inv = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours_inv, _ = find_contours(binary_inv)
                comps = []
                for c in contours_inv:
                    a = cv2.contourArea(c)
                    if a < max(20, (h*w)//500):
                        continue
                    x,y,ww,hh = cv2.boundingRect(c)
                    comps.append((x,ww))
                # sort by x and check separation
                if len(comps) >= 2:
                    comps.sort(key=lambda x: x[0])
                    first_x, first_w = comps[0]
                    second_x, second_w = comps[1]
                    gap = second_x - (first_x + first_w)
                    if gap > w * 0.20:
                        return '10', max(top_info['score'], 0.75)
                if score_diff > 0.2:
                    return top_name, top_info['score']
            
            # Heurística para distinguir 3 vs 2
            if top_name in ('3','2') or name in ('3','2'):
                # Use right/left pixel distribution in inverted binary
                _, binary_inv = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                mid = binary_inv.shape[1] // 2
                left_sum = np.sum(binary_inv[:, :mid] > 0)
                right_sum = np.sum(binary_inv[:, mid:] > 0)
                if right_sum > 1.3 * left_sum:
                    return '3', max(top_info['score'], 0.7)
                # 2 tends to have a stronger top horizontal stroke and right-bottom diagonal
                top_strip = rank_symbol_enhanced[:max(2, h//6), :]
                top_strength = np.sum(top_strip > 0)
                total = np.sum(rank_symbol_enhanced > 0) + 1e-6
                top_ratio = top_strength / total
                # If top horizontal significant -> bias to '2'
                if top_ratio > 0.18:
                    return '2', max(top_info['score'], 0.68)
                if score_diff > 0.15:
                    return top_name, top_info['score']
            
            # Otherwise, if best candidate much better, take it
            if score_diff > 0.20 or top_info['score'] > 0.72:
                return top_name, top_info['score']
    
    # As final fallback, if low score but shape metrics give hints:
    if score < 0.55:
        # If shape shows many holes -> 8
        if shape_metrics["defects"] >= 2 and shape_metrics["circularity"] > 0.4:
            return '8', max(score, 0.65)
        # If aspect ratio narrow and vertical strokes -> likely '1' but we don't alter here
        # If vertex count small and radial uniformity (approx) maybe '2'/'3' decisions already tried
    return name, score