import os
import cv2
from utils import load_image_rgb
from contours import find_card_contour_from_binary
from transforms import four_point_transform, extract_top_left_corner
from symbols import extract_symbols_from_corner

def process_card_image(image_path, visualize=False):
    try:
        image_rgb = load_image_rgb(image_path)
    except FileNotFoundError as e:
        print(e)
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    card_contour, area = find_card_contour_from_binary(binary, min_area=10000)
    if card_contour is None:
        print(f"No se encontró contorno de carta en {image_path}.")
        return None
    warped = four_point_transform(image_rgb, card_contour, width=300, height=420)
    corner = extract_top_left_corner(warped)
    thresh_corner, symbols = extract_symbols_from_corner(corner)
    return {
        "image_path": image_path,
        "original": image_rgb,
        "binary": binary,
        "card_contour": card_contour,
        "warped": warped,
        "corner": corner,
        "thresh_corner": thresh_corner,
        "symbols": symbols
    }