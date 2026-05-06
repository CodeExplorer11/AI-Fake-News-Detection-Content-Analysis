# =====================================
# AI Fake News Generator & Detector
# Professional Brown Theme Version
# =====================================

import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from fpdf import FPDF
import datetime
import os

# ============================
# DEVICE SETUP
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# LOAD GPT-2 MODEL
# ============================

gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
gen_model.to(device)

# ============================
# LOAD FAKE NEWS DETECTION MODEL
# ============================

detector = pipeline(
    "text-classification",
    model="jy46604790/Fake-News-Bert-Detect",
    tokenizer="jy46604790/Fake-News-Bert-Detect",
    device=0 if torch.cuda.is_available() else -1
)

# ============================
# REPORT GENERATION
# ============================

def save_report(text, label, confidence):

    filename = "report.pdf"

    pdf = FPDF()

    # Check if file exists
    if os.path.exists(filename):

        # Load existing content by recreating new page
        pdf.add_page()

    else:
        pdf.add_page()
        pdf.set_font("Arial", size=18)
        pdf.cell(200, 10, txt="Fake News Detection Report", ln=True)
        pdf.ln(5)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now()}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)

    pdf.multi_cell(0, 10, txt=f"Text: {text}")
    pdf.ln(10)

    pdf.output(filename)

    return os.path.abspath(filename)
# ============================
# GENERATE NEWS
# ============================

def generate_news(prompt):

    if prompt.strip() == "":
        return "Please enter a valid prompt."

    input_ids = gen_tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = gen_model.generate(
        input_ids,
        max_length=200,
        temperature=0.9,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    news = gen_tokenizer.decode(output[0], skip_special_tokens=True)

    return news

# ============================
# DETECT NEWS
# ============================

def detect_news(text):

    try:

        if text.strip() == "":
            return "Enter valid text", 0, None

        result = detector(text)[0]

        confidence = float(result['score'])

        if "FAKE" in result['label'].upper():
            label = "FAKE NEWS"
        else:
            label = "REAL NEWS"

        filename = save_report(text, label, confidence)

        return label, confidence, filename

    except Exception as e:

        print(e)
        return "Detection Failed", 0, None

# ============================
# CLEAR FUNCTIONS
# ============================

def clear_generate():
    return ""

def clear_detect():
    return "", 0, None

# ============================
# BROWN THEME CSS
# ============================

brown_css = """

body {
    background-color: #3b2f2f;
}

.gradio-container {
    background-color: #4e342e !important;
    font-size: 20px !important;
}

h1 {
    color: #d7ccc8 !important;
    font-size: 40px !important;
}

h3 {
    color: #efebe9 !important;
    font-size: 24px !important;
}

label {
    font-size: 20px !important;
    color: #fff3e0 !important;
}

textarea, input {
    font-size: 18px !important;
}

button {
    background-color: #6d4c41 !important;
    color: white !important;
    font-size: 20px !important;
    border-radius: 10px !important;
}

"""

# ============================
# GUI
# ============================

with gr.Blocks(css=brown_css, theme=gr.themes.Base()) as app:

    gr.Markdown("# Fake News Generator & Detector")
    gr.Markdown("### Built using GPT-2 and BERT")

    with gr.Tab("Generate News"):

        prompt = gr.Textbox(
            label="Enter Prompt",
            lines=3
        )

        generate_btn = gr.Button("Generate")

        output_news = gr.Textbox(
            label="Generated News",
            lines=10
        )

        clear_btn1 = gr.Button("Clear")

        generate_btn.click(
            generate_news,
            inputs=prompt,
            outputs=output_news
        )

        clear_btn1.click(
            clear_generate,
            outputs=output_news
        )

    with gr.Tab("Detect News"):

        input_text = gr.Textbox(
            label="Enter News Text",
            lines=5
        )

        detect_btn = gr.Button("Detect")

        prediction = gr.Textbox(label="Prediction")

        confidence = gr.Slider(
            minimum=0,
            maximum=1,
            label="Confidence"
        )

        report = gr.File(label="Download Report")

        clear_btn2 = gr.Button("Clear")

        detect_btn.click(
            detect_news,
            inputs=input_text,
            outputs=[prediction, confidence, report]
        )

        clear_btn2.click(
            clear_detect,
            outputs=[prediction, confidence, report]
        )

# ============================
# RUN
# ============================

app.launch()