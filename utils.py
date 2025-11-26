import cv2
import matplotlib.pyplot as plt

def show_img(img, title="", figsize=(6, 6), cmap=None, mode="block"):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if len(img.shape) == 2 or cmap is not None:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    if mode == "block":
        plt.show()
    elif mode == "enter":
        plt.show(block=False)
        try:
            input("Presiona Enter para continuar...")
        except EOFError:
            pass
        plt.close(fig)
    elif mode == "auto":
        plt.show(block=False)
        plt.pause(0.25)
        plt.close(fig)
    else:
        plt.show()

def wait_enter(enabled=True, message="Presiona Enter para continuar..."):
    if enabled:
        try:
            input(message)
        except EOFError:
            pass

def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)