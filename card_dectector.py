"""
Card detector: detects playing card color (red/black), rank (A,2..10,J,Q,K) and suit (hearts,diamonds,clubs,spades)
from a live external camera using OpenCV, pytesseract and template matching for suits.

Requirements:
  - Python 3.8+
  - OpenCV: pip install opencv-python
  - pytesseract: pip install pytesseract
  - Tesseract OCR installed on your system (https://github.com/tesseract-ocr/tesseract)
  - numpy

How it works (brief):
  1. Grab frame from camera (device index configurable).
  2. Detect largest rectangular contour (assumed card), do perspective transform (warp).
  3. Extract top-left corner region (rank + suit) and run OCR for rank.
  4. Use template-matching against suit templates to recognize the suit.
  5. Determine color by checking for red hues in the suit region.

Notes:
  - Put suit templates (png) in ./templates/suits/ named: hearts.png, diamonds.png, clubs.png, spades.png
  - Optionally add rank templates for more robust detection (not included here).
  - Camera index defaults to 0 but you can pass a different index when calling the script.

Run:
  python card_detector.py --cam 1

"""

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import argparse
import os

# --- User-configurable ---
SUIT_TEMPLATES_DIR = os.path.join('templates', 'suits')
CARD_MAX_AREA = 120000  # tune depending on resolution/distance
CARD_MIN_AREA = 5000
OCR_CONFIG = '--psm 10 -c tessedit_char_whitelist=A23456789TJQK'  # single char recognition

# If pytesseract can't be found, set pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_tesseract>'
# e.g. pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def find_first_camera(max_index=8):
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return i
    return None

def open_camera(cam_arg):
    # cam_arg puede ser int (índice) o string (nombre). Si es None, intenta auto-detectar.
    if cam_arg is None:
        idx = find_first_camera(10)
        if idx is None:
            raise RuntimeError('No se encontró ninguna cámara disponible.')
        print('Usando cámara índice:', idx)
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    # intenta parsear como int
    try:
        idx = int(cam_arg)
        print('Abriendo cámara por índice:', idx)
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except ValueError:
        # tratar cam_arg como nombre de dispositivo (Windows DirectShow)
        ds_name = f'video={cam_arg}'
        print('Abriendo cámara por nombre (DirectShow):', ds_name)
        return cv2.VideoCapture(ds_name, cv2.CAP_DSHOW)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', help='índice (0,1,...) o nombre dispositivo (ej: "Reincubate Camo"). Si se omite detecta automáticamente.')
    args = parser.parse_args()

    cap = open_camera(args.cam)
    if not cap.isOpened():
        raise SystemExit('No se pudo abrir la cámara. Revisa que Camo esté en USB mode y que ninguna otra app la esté usando.')
    ret, frame = cap.read()
    print('Frame OK?', ret, 'shape:', None if frame is None else frame.shape)
    cap.release()


def load_suit_templates(folder):
    templates = {}
    for name in ['hearts', 'diamonds', 'clubs', 'spades']:
        path = os.path.join(folder, f'{name}.png')
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # convert to gray
            if img.shape[-1] == 4:
                # handle alpha
                alpha = img[:, :, 3]
                bgr = img[:, :, :3]
                mask = (alpha > 0).astype(np.uint8) * 255
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray[mask == 0] = 255
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            templates[name] = gray
    return templates


def preprocess_frame_for_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def find_card_contour(frame):
    thresh = preprocess_frame_for_contours(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < CARD_MIN_AREA or area > CARD_MAX_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    return best


def order_points(pts):
    # pts: 4x1x2 -> reshape
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def extract_rank_and_suit(card_img):
    # assume card_img in BGR, size variable. extract top-left region where rank+small suit are printed
    h, w = card_img.shape[:2]
    # ROI cropping ratios, may need tuning depending on card design
    roi = card_img[int(0.02 * h):int(0.25 * h), int(0.02 * w):int(0.22 * w)]
    return roi


def ocr_rank(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # enlarge to help OCR
    scale = max(1, int(200.0 / max(roi.shape[:2])))
    gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # try OCR
    text = pytesseract.image_to_string(thresh, config=OCR_CONFIG)
    text = text.strip().upper()
    # normalize common variants
    text = text.replace('0', '10')
    text = text.replace('T', '10')
    # allowed ranks
    ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    # sometimes pytesseract returns multiple chars, pick the best
    for ch in text.split()[:1]:
        if ch in ranks:
            return ch
    # fallback: look for any char in string
    for c in text:
        if c in ''.join(ranks):
            return c
    return None


def match_suit(roi_gray, templates):
    # roi_gray: grayscale image, templates: dict name->gray image
    best_name = None
    best_val = 0
    # small ROI might contain both rank and suit; search with resizing
    for name, tmpl in templates.items():
        # try multiple scales of template matching
        for scale in np.linspace(0.5, 1.5, 11):
            th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
            if th < 10 or tw < 10 or th > roi_gray.shape[0] or tw > roi_gray.shape[1]:
                continue
            resized_t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi_gray, resized_t, cv2.TM_CCOEFF_NORMED)
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
            if maxv > best_val:
                best_val = maxv
                best_name = name
    # threshold match
    if best_val > 0.45:
        return best_name, best_val
    return None, best_val


def detect_color_from_suit_region(roi_bgr):
    # We'll decide red vs black by testing if red pixels are present in suit region.
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # red has two hue ranges in OpenCV (0-10) and (160-180)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_ratio = (cv2.countNonZero(mask) / (roi_bgr.shape[0]*roi_bgr.shape[1]))
    if red_ratio > 0.005:  # threshold; adjust as necessary
        return 'red', red_ratio
    else:
        return 'black', red_ratio


def main(camera_index=0):
    templates = load_suit_templates(SUIT_TEMPLATES_DIR)
    if not templates:
        print('Warning: no suit templates loaded. Put heart/diamond/club/spade PNGs in templates/suits/')

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'Could not open camera index {camera_index}. Try a different index.')
        return

    print('Press q to quit, s to save last detected card image (for template collection).')
    saved_count = 0
    last_card = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame.copy()
        card_contour = find_card_contour(frame)
        label = 'No card'
        if card_contour is not None:
            try:
                warped = four_point_transform(orig, card_contour)
                last_card = warped.copy()
                # standardize orientation by ensuring width > height
                if warped.shape[0] > warped.shape[1]:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                roi = extract_rank_and_suit(warped)
                rank = ocr_rank(roi)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                suit_name, match_val = match_suit(roi_gray, templates)
                color, red_ratio = detect_color_from_suit_region(roi)
                label = f'{rank or "?"} of {suit_name or "?"} ({color})'
                # draw warped image small on original for feedback
                h,w = warped.shape[:2]
                scale = 200 / max(h,w)
                small = cv2.resize(warped, (int(w*scale), int(h*scale)))
                x_offset = 10
                frame[10:10+small.shape[0], x_offset:x_offset+small.shape[1]] = small
            except Exception as e:
                label = 'Error processing card'
                print('Processing error:', e)

            # draw contour
            cv2.drawContours(frame, [card_contour], -1, (0,255,0), 2)

        cv2.putText(frame, label, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow('Card detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and last_card is not None:
            # save last_card to disk for template collection
            os.makedirs('captures', exist_ok=True)
            fname = os.path.join('captures', f'card_{saved_count}.png')
            cv2.imwrite(fname, last_card)
            print('Saved', fname)
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='camera index (0,1,...)')
    args = parser.parse_args()
    main(camera_index=args.cam)
