from agents import function_tool
from PIL import Image, ImageOps, ImageFilter
from tools.utils import read_images, set_images
import numpy as np

SUPPORTED_FILTERS = ("grayscale", "sepia", "invert", "blur", "sharpen", "edge_enhance")


def _apply(img: Image.Image, filter_name: str) -> Image.Image:
    if filter_name == "grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    elif filter_name == "sepia":
        arr = np.array(img, dtype=np.float64)
        r = (arr[..., 0] * 0.393 + arr[..., 1] * 0.769 + arr[..., 2] * 0.189).clip(0, 255)
        g = (arr[..., 0] * 0.349 + arr[..., 1] * 0.686 + arr[..., 2] * 0.168).clip(0, 255)
        b = (arr[..., 0] * 0.272 + arr[..., 1] * 0.534 + arr[..., 2] * 0.131).clip(0, 255)
        return Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8))
    elif filter_name == "invert":
        return ImageOps.invert(img)
    elif filter_name == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_name == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    elif filter_name == "edge_enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)


@function_tool
def apply_filter(filter_name: str) -> dict:
    """
    Apply a filter to both images in memory and update them for subsequent similarity calculations.

    Args:
        filter_name: Filter to apply. Supported: grayscale, sepia, invert, blur, sharpen, edge_enhance.

    Returns:
        dict: {"success": bool, "error": str | None}
    """
    if filter_name not in SUPPORTED_FILTERS:
        return {"success": False, "error": f"Unsupported filter '{filter_name}'. Choose from: {', '.join(SUPPORTED_FILTERS)}"}

    image1, image2 = read_images()
    set_images(_apply(image1.convert("RGB"), filter_name), _apply(image2.convert("RGB"), filter_name))
    return {"success": True, "error": None}
