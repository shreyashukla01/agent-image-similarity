# Image Similarity Agent

An agentic image similarity analyzer built with the OpenAI Agents SDK and GPT-4.1-mini. Upload two images through a Gradio UI and get a structured similarity analysis across three metrics.

## How it works

1. Upload two images in the UI — they are loaded into memory.
2. The agent calls three similarity tools:
   - **Cosine similarity** — pixel-level similarity, sensitive to color and structure
   - **CLIP similarity** — semantic similarity using `openai/clip-vit-base-patch16`
   - **SSIM** — structural similarity (luminance, contrast, structure)
3. If the images are significantly different (cosine or SSIM below 0.7), the agent may apply a filter to both images to normalize them before recalculating.
4. Results are returned with scores, plain-language interpretations, filters applied, and an overall conclusion.

## Setup

Add your OpenAI API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Opens at `http://localhost:7860`.

## Project structure

```
agent/
  similarity_agent.py          # Agent definition, model, tools, and instructions
  structured_model_output.py   # Pydantic output schema
tools/
  similarity_tools.py          # Cosine, CLIP, and SSIM similarity tools
  image_modification_tools.py  # apply_filter tool
  utils.py                     # In-memory image store (set_images / read_images)
app.py                         # Gradio UI
```

## Available filters

`grayscale`, `sepia`, `invert`, `blur`, `sharpen`, `edge_enhance`


![Similarity Agent Gradio UI](<Similarity agent UI.png>)