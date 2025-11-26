import numpy as np
import cv2
import math

def order_points(pts):
    """
    Ordena los puntos en el orden: top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    
    # Calcular el centro
    center = np.mean(pts, axis=0)
    
    # Calcular ángulos desde el centro para cada punto
    angles = []
    for pt in pts:
        angle = math.atan2(pt[1] - center[1], pt[0] - center[0])
        angles.append((angle, pt))
    
    # Ordenar por ángulo (comenzando desde arriba-izquierda, sentido horario)
    angles.sort(key=lambda x: x[0])
    
    # Asignar puntos en orden: top-left, top-right, bottom-right, bottom-left
    sorted_points = [pt for angle, pt in angles]
    
    # Encontrar los puntos más arriba y más abajo
    y_coords = [pt[1] for pt in sorted_points]
    top_indices = sorted(range(4), key=lambda i: y_coords[i])[:2]  # Los 2 puntos más arriba
    bottom_indices = sorted(range(4), key=lambda i: y_coords[i])[2:]  # Los 2 puntos más abajo
    
    top_points = [sorted_points[i] for i in top_indices]
    if top_points[0][0] < top_points[1][0]:
        rect[0] = top_points[0]
        rect[1] = top_points[1]
    else:
        rect[0] = top_points[1]
        rect[1] = top_points[0]
    
    bottom_points = [sorted_points[i] for i in bottom_indices]
    if bottom_points[0][0] < bottom_points[1][0]:
        rect[3] = bottom_points[0]
        rect[2] = bottom_points[1]
    else:
        rect[3] = bottom_points[1]
        rect[2] = bottom_points[0]
    
    return rect

def four_point_transform(image_rgb, pts, width=300, height=420):
    """
    Aplica transformación perspectiva asegurando que la carta quede en orientación vertical
    con proporciones correctas
    """
    src = pts.reshape(4, 2)
    src_ord = order_points(src)
    
    # Calcular las dimensiones de los lados de la carta detectada
    # Lado superior
    top_width = np.linalg.norm(src_ord[1] - src_ord[0])
    # Lado izquierdo  
    left_height = np.linalg.norm(src_ord[3] - src_ord[0])
    # Lado inferior
    bottom_width = np.linalg.norm(src_ord[2] - src_ord[3])
    # Lado derecho
    right_height = np.linalg.norm(src_ord[2] - src_ord[1])
    
    # Calcular las dimensiones promedio
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    print(f"Dimensiones detectadas: ancho={avg_width:.1f}, alto={avg_height:.1f}, ratio={avg_height/avg_width:.2f}")
    
    # Determinar si la carta está en orientación horizontal (más ancha que alta)
    card_is_horizontal = avg_width > avg_height
    
    if card_is_horizontal:
        print("Carta detectada en orientación horizontal - rotando a vertical")
        actual_card_width = avg_height
        actual_card_height = avg_width
        
        src_ord_rotated = np.array([
            src_ord[1],
            src_ord[2],
            src_ord[3],
            src_ord[0]
        ], dtype="float32")
        src_ord = src_ord_rotated
    else:
        print("Carta detectada en orientación vertical - manteniendo orientación")
        actual_card_width = avg_width
        actual_card_height = avg_height
    
    target_aspect_ratio = 1.4  # Ratio estándar de cartas de póker (altura/ancho)
    detected_ratio = actual_card_height / actual_card_width
    
    if 1.2 <= detected_ratio <= 1.8:
        if actual_card_width > actual_card_height:
            actual_card_width, actual_card_height = actual_card_height, actual_card_width
        
        scale_factor = min(width / actual_card_width, height / actual_card_height)
        final_width = int(actual_card_width * scale_factor)
        final_height = int(actual_card_height * scale_factor)
        
        final_width = min(final_width, width)
        final_height = min(final_height, height)
        
    else:
        print(f"Ratio detectado {detected_ratio:.2f} fuera de rango esperado, usando dimensiones por defecto")
        final_width = width
        final_height = height
    
    print(f"Dimensiones finales: {final_width}x{final_height}, ratio={final_height/final_width:.2f}")
    
    dst = np.array([
        [0, 0],
        [final_width-1, 0],
        [final_width-1, final_height-1],
        [0, final_height-1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_ord, dst)
    warped = cv2.warpPerspective(image_rgb, M, (final_width, final_height))
    
    return warped

def extract_top_left_corner(warped_card, w_ratio=0.28, h_ratio=0.40):
    """
    Extrae la esquina superior izquierda con proporciones ajustadas
    """
    h, w = warped_card.shape[:2]
    
    if w < 250:
        w_ratio = min(w_ratio * 1.2, 0.35)
        h_ratio = min(h_ratio * 1.2, 0.45)
    
    rw = int(w * w_ratio)
    rh = int(h * h_ratio)
    
    rw = max(rw, 60)
    rh = max(rh, 80)
    
    rw = min(rw, w)
    rh = min(rh, h)
    
    print(f"Esquina extraída: {rw}x{rh} de carta {w}x{h}")
    
    return warped_card[0:rh, 0:rw].copy()