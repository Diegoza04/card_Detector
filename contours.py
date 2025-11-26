import cv2
import numpy as np

def find_contours(binary):
    res = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    return contours, hierarchy

def find_card_contour_from_binary(binary, min_area=10000):
    contours, _ = find_contours(binary)
    max_area = 0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            best = approx
    return best, max_area

def find_all_card_contours_from_binary(binary, min_area=10000):
    """
    Encuentra TODOS los contornos que parezcan cartas (polígonos de 4 puntos con área suficiente).
    Devuelve lista de (approx_contour, area).
    """
    contours, _ = find_contours(binary)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) != 4:
            continue
        # Validar razón de aspecto (altura/ancho) aproximada de una carta
        x,y,w,h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        ratio = h / w
        # Cartas verticales ~1.3–1.5, horizontales ~0.65–0.75 (permitimos rango más amplio)
        if not (0.55 <= ratio <= 1.9):
            continue
        candidates.append((approx, area))
    # Opcional: ordenar por área (mayor a menor)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates