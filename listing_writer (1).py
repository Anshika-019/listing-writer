"""
Listing Writer: turns product details (and an optional photo) into a
ready-to-paste Etsy/eBay listing, using Gemma running locally through Ollama.

Setup:
  1. Install Ollama from https://ollama.com
  2. ollama pull gemma3:4b
  3. pip install "gradio<6" ollama
  4. python listing_writer.py
"""

import gradio as gr
import ollama

MODEL = "gemma3:4b"  # swap for a bigger/newer Gemma if your machine can handle it

SYSTEM_PROMPT = """You write product listings for small online sellers on Etsy and eBay.

STRICT RULES:
- Only use facts the seller gave you or that are clearly visible in the photo.
- Never add features, accessories, specs or "what's in the box" items that weren't given.
- Never write as if you personally used the product. No "I love", "I've been impressed".
- Start directly with "TITLE:". No intro line, no closing line.

Reply in exactly this format:

TITLE:
<one title under 140 characters, front-loaded with search keywords>

DESCRIPTION:
<2 short paragraphs, then a bullet list of only the given details>

TAGS:
<13 comma-separated search tags, each under 20 characters>
"""


def write_listing(name, details, platform, tone, photo):
    if not name.strip():
        return "Add a product name to get started."

    prompt = (
        f"Platform: {platform}\n"
        f"Tone: {tone}\n"
        f"Product: {name}\n"
        f"Seller's notes: {details or 'none'}\n"
    )
    if photo:
        prompt += "A product photo is attached. Use what you can see in it.\n"

    message = {"role": "user", "content": prompt}
    if photo:
        message["images"] = [photo]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, message],
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            f"Couldn't reach the model ({e}).\n\n"
            f"Check that Ollama is running and that you've run: ollama pull {MODEL}"
        )


with gr.Blocks(title="Listing Writer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# Listing Writer\n"
        "Describe your product and get a title, description and tags you can paste "
        "straight into your shop. Powered by Gemma, a free open-weight model, "
        "so there's no paid API behind it."
    )
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Product name", placeholder="Hand-thrown ceramic mug")
            details = gr.Textbox(
                label="Details",
                lines=5,
                placeholder="Speckled glaze, holds 12 oz, dishwasher safe, made in Ohio",
            )
            photo = gr.Image(label="Photo (optional)", type="filepath")
            with gr.Row():
                platform = gr.Radio(["Etsy", "eBay"], value="Etsy", label="Platform")
                tone = gr.Radio(
                    ["Friendly", "Professional", "Playful"], value="Friendly", label="Tone"
                )
            button = gr.Button("Write listing", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Your listing", lines=22, show_copy_button=True)

    button.click(write_listing, [name, details, platform, tone, photo], output)

if __name__ == "__main__":
    app.launch()
