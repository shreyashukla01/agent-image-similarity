from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
from agents import Runner
from agent.similarity_agent import similarity_agent
from agent.structured_model_output import SimilarityOutput
from tools.utils import set_images


async def compare_images(img1: Image.Image, img2: Image.Image) -> str:
    if img1 is None or img2 is None:
        return "Please upload both images."

    set_images(img1, img2)

    result = await Runner.run(similarity_agent, "Analyze the two images.")
    output: SimilarityOutput = result.final_output

    filters = ', '.join(output.filters_applied) if output.filters_applied else 'None'
    return f"""
        ### Cosine Similarity - `{output.cosine_similarity_score}`
        {output.cosine_similarity_interpretation}

        ### CLIP Similarity - `{output.clip_similarity_score}`
        {output.clip_similarity_interpretation}

        ### SSIM - `{output.ssim_similarity_score}`
        {output.ssim_similarity_interpretation}

        ---
        **Filters applied:** {filters}

        ### Conclusion
        {output.overall_conclusion}
    """


with gr.Blocks(title="Image Similarity Agent") as demo:
    gr.Markdown("## Image Similarity Agent")
    with gr.Row():
        img1 = gr.Image(label="Image 1", type="pil", sources=["upload"])
        img2 = gr.Image(label="Image 2", type="pil", sources=["upload"])
    btn = gr.Button("Compare Images", variant="primary")
    output = gr.Markdown()
    btn.click(compare_images, inputs=[img1, img2], outputs=output)

demo.launch()
