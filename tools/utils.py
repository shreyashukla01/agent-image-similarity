from PIL import Image, ImageOps

_state: dict = {"images": None}


def set_images(img1: Image.Image, img2: Image.Image) -> None:
    img1 = ImageOps.exif_transpose(img1)
    img2 = ImageOps.exif_transpose(img2)
    print(f"Images set: {img1.size}, {img2.size}")
    _state["images"] = (img1.copy(), img2.copy())


def read_images() -> tuple:
    if _state["images"] is None:
        raise RuntimeError("No images loaded. Upload images before running analysis.")
    img1, img2 = _state["images"]
    print(f"[DEBUG read_images] img1 size={img1.size} pixel[0,0]={img1.getpixel((0,0))}, img2 size={img2.size} pixel[0,0]={img2.getpixel((0,0))}")
    return _state["images"]
