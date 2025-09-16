import gradio as gr
from PIL import Image

from inference import infer


# Example processing function
def process_image_text(image: Image.Image, text: str) -> str:
    # Dummy processing - Replace with your actual logic
    #return f"Received text: '{text}' and an image of size {image.size}"
    return infer (image, text)


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Cell Image and Text prompt")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload your cell image")
        text_input = gr.Textbox(label="Enter Your Prompt")

    output_text = gr.Textbox(label="Output")

    submit_btn = gr.Button("Submit")

    submit_btn.click(fn=process_image_text, inputs=[image_input, text_input], outputs=output_text)

# Launch the app
demo.launch()
