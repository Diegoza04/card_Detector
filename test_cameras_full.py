# test_cameras_full.py
import cv2
import time

def test_indices(max_index=10):
    print("Probando índices...")
    found = []
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        print(f"Index {i}: opened={cap.isOpened()} frame_ok={ok} shape={None if frame is None else frame.shape}")
        if ok:
            found.append(("index", i))
        cap.release()
        time.sleep(0.2)
    return found

def test_names(names):
    print("\nProbando nombres DirectShow candidatos...")
    found = []
    for name in names:
        ds = f"video={name}"
        cap = cv2.VideoCapture(ds, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        print(f"Name '{ds}': opened={cap.isOpened()} frame_ok={ok} shape={None if frame is None else frame.shape}")
        if ok:
            found.append(("name", name))
        cap.release()
        time.sleep(0.2)
    return found

if __name__ == "__main__":
    candidates = ["Reincubate Camo", "Camo", "Camo Studio", "Camo Camera", "Camo Virtual", "OBS Virtual Camera", "OBS-Camera", "Camo (Virtual)"]
    found = test_indices(10) + test_names(candidates)
    print("\nEncontrados (usable):", found)
