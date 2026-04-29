from agents import Agent
from agent.structured_model_output import SimilarityOutput
from tools.similarity_tools import (
    calculate_cosine_similarity,
    calculate_clip_similarity,
    calculate_ssim_similarity,
)
from tools.image_modification_tools import apply_filter

similarity_agent = Agent(
    name="Image Similarity Agent",
    model="gpt-4.1-mini",
    output_type=SimilarityOutput,
    instructions="""
        You are an image similarity analyst. Two images are always preloaded - the tools read them automatically, no image names or paths needed.

        Analyze how similar the two images are using these tools:
        - calculate_cosine_similarity: Pixel-level similarity, sensitive to color and structure.
        - calculate_clip_similarity: Semantic similarity via CLIP, capturing high-level concepts.
        - calculate_ssim_similarity: Structural similarity.

        Follow this process:

        1. YOU MUST call all three similarity tools. Do not skip any.

        2. Check the scores:
        - If cosine_similarity_score < 0.7 OR ssim_similarity_score < 0.7, you MUST apply a filter.
        - Choose the most appropriate filter (e.g. grayscale removes color bias, blur reduces noise).
        - Call apply_filter with the filter_name - it updates both images in memory automatically.
        - Available filters: grayscale, sepia, invert, blur, sharpen, edge_enhance.
        - After applying the filter, recalculate all three scores.

        3. Return the exact scores from the tool results. DO NOT modify or estimate any score values.
        Provide plain-language interpretations and an overall conclusion.
    """,
    tools=[
        calculate_cosine_similarity,
        calculate_clip_similarity,
        calculate_ssim_similarity,
        apply_filter,
    ],
)
