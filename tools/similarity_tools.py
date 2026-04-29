from agents import function_tool
from PIL import ImageOps
from tools.utils import read_images
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from transformers import CLIPProcessor, CLIPModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("Loading CLIP model (openai/clip-vit-base-patch16)...")
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
_clip_model.eval()
logger.info("CLIP model loaded.")


@function_tool
def calculate_cosine_similarity() -> dict:
    """Calculate pixel-level cosine similarity between the two uploaded images."""
    try:
        image1, image2 = read_images()
        image1 = ImageOps.grayscale(image1.resize((256, 256)))
        image2 = ImageOps.grayscale(image2.resize((256, 256)))
        vec1 = np.asarray(image1, dtype=np.float32).flatten()
        vec2 = np.asarray(image2, dtype=np.float32).flatten()
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return {"success": False, "cosine_similarity_score": None, "error": "One or both images are empty"}
        score = float(np.dot(vec1 / norm1, vec2 / norm2))
        logger.info(f"Cosine similarity: {score:.4f}")
        return {"success": True, "cosine_similarity_score": round(score, 4), "error": None}
    except Exception as e:
        logger.error(f"Cosine similarity failed: {e}")
        return {"success": False, "cosine_similarity_score": None, "error": str(e)}


@function_tool
def calculate_clip_similarity() -> dict:
    """Calculate semantic CLIP similarity between the two uploaded images."""
    try:
        image1, image2 = read_images()
        inputs1 = _clip_processor(images=image1.convert("RGB"), return_tensors="pt")
        inputs2 = _clip_processor(images=image2.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            emb1 = _clip_model.get_image_features(**inputs1)
            emb2 = _clip_model.get_image_features(**inputs2)
        if hasattr(emb1, "pooler_output"):
            emb1, emb2 = emb1.pooler_output, emb2.pooler_output
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        score = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        logger.info(f"CLIP similarity: {score:.4f}")
        return {"success": True, "clip_similarity_score": round(float(score), 4), "error": None}
    except Exception as e:
        logger.error(f"CLIP similarity failed: {e}")
        return {"success": False, "clip_similarity_score": None, "error": str(e)}


@function_tool
def calculate_ssim_similarity() -> dict:
    """Calculate structural similarity (SSIM) between the two uploaded images."""
    try:
        image1, image2 = read_images()
        arr1 = np.asarray(image1.convert("L").resize((256, 256)), dtype=np.float32)
        arr2 = np.asarray(image2.convert("L").resize((256, 256)), dtype=np.float32)
        score = ssim(arr1, arr2, data_range=255)
        logger.info(f"SSIM similarity: {score:.4f}")
        return {"success": True, "ssim_similarity_score": round(float(score), 4), "error": None}
    except Exception as e:
        logger.error(f"SSIM similarity failed: {e}")
        return {"success": False, "ssim_similarity_score": None, "error": str(e)}
