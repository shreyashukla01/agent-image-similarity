from pydantic import BaseModel, Field


class SimilarityOutput(BaseModel):
    cosine_similarity_score: float = Field(
        description="Cosine similarity score between the two images."
    )
    cosine_similarity_interpretation: str = Field(
        description="Interpretation of the cosine similarity score."
    )
    clip_similarity_score: float = Field(
        description="CLIP similarity score between the two images."
    )
    clip_similarity_interpretation: str = Field(
        description="Interpretation of the CLIP similarity score."
    )
    ssim_similarity_score: float = Field(
        description="SSIM similarity score between the two images."
    )
    ssim_similarity_interpretation: str = Field(
        description="Interpretation of the SSIM similarity score."
    )
    filters_applied: list[str] = Field(
        description="List of filters applied to the images during analysis."
    )
    overall_conclusion: str = Field(
        description="Overall conclusion on how similar the images are."
    )
