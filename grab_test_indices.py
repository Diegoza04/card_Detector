import cv2
import os

os.makedirs('captures', exist_ok=True)
indices = [0,1,2]  # los encontrados por tu script

for i in indices:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    if ok and frame is not None:
        fname = f'captures/cam_{i}.png'
        cv2.imwrite(fname, frame)
        print(f'Guardado {fname} (shape={frame.shape})')
    else:
        print(f'No se pudo leer índice {i}')
    cap.release()
