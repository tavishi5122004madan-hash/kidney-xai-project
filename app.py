
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLBACKEND"] = "Agg"

import gc
import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


# ==================================================
# 1. MODEL CONFIGURATION
# ==================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "MODELS",
    "kidney_xai_model_v2.keras"
)

CLASS_NAMES = ["Normal", "Tumor"]

tf.config.set_visible_devices([], "GPU")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

augmentation = model.get_layer("data_augmentation")
backbone = model.get_layer("efficientnetb0")
pooling = model.get_layer("global_average_pooling2d")
dropout = model.get_layer("dropout")
classifier = model.get_layer("dense")


# ==================================================
# 2. CLINICAL WHITE AND TEAL THEME
# ==================================================

CSS = """
:root {
    --teal: #0f766e;
    --teal-dark: #115e59;
    --teal-light: #ccfbf1;
    --navy: #17324d;
    --muted: #475569;
    --border: #d5e3e7;
    --surface: #ffffff;
    --background: #f2f7f8;
}

/* Main page */
.gradio-container {
    background: #f2f7f8 !important;
    color: #17324d !important;
    font-family: Arial, sans-serif !important;
}

.gradio-container .main,
.gradio-container .wrap {
    background: transparent !important;
}

/* General text */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label {
    color: #17324d !important;
}

.gradio-container p {
    color: #475569;
}

/* Header */
.main-header {
    background: linear-gradient(
        120deg,
        #ffffff 0%,
        #e0f5f1 100%
    ) !important;
    border: 1px solid #c8e4df;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 20px;
}

.eyebrow {
    color: #0f766e !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.main-header h1 {
    color: #17324d !important;
    font-size: clamp(27px, 4vw, 40px);
    margin: 10px 0;
}

.main-header p {
    color: #475569 !important;
    line-height: 1.7;
    max-width: 760px;
}

.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    background: #ccfbf1 !important;
    color: #115e59 !important;
    font-size: 12px;
    font-weight: 700;
}

/* Section headings */
.section-heading {
    color: #115e59 !important;
    font-size: 21px;
    font-weight: 750;
    margin: 14px 0 8px;
}

.section-subtitle {
    color: #475569 !important;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 12px;
}

/* Panels */
.panel {
    background: #ffffff !important;
    border: 1px solid #d5e3e7 !important;
    border-radius: 15px;
    padding: 18px;
    color: #17324d !important;
}

.panel p,
.panel strong {
    color: #17324d !important;
}

/* Probability cards */
.probability-card {
    background: #ffffff !important;
    border: 1px solid #d5e3e7 !important;
    border-radius: 14px;
    padding: 17px;
    margin: 10px 0;
}

.probability-heading {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: #17324d !important;
    font-weight: 700;
    margin-bottom: 10px;
}

.probability-heading span {
    color: #17324d !important;
}

.probability-track {
    height: 12px;
    border-radius: 20px;
    background: #e2e8f0 !important;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 20px;
}

.normal-fill {
    background: #0d9488 !important;
}

.tumor-fill {
    background: #d97757 !important;
}

/* Information boxes */
.info-box {
    background: #ecfdf5 !important;
    border: 1px solid #b7e4d7;
    border-left: 4px solid #0f766e;
    border-radius: 12px;
    padding: 16px;
    color: #17324d !important;
    line-height: 1.7;
    margin-top: 15px;
}

.info-box strong {
    color: #115e59 !important;
}

.warning-box {
    background: #fff8e8 !important;
    border: 1px solid #f3dfad;
    border-left: 4px solid #d97706;
    border-radius: 12px;
    padding: 18px;
    color: #854d0e !important;
    line-height: 1.7;
}

.warning-box strong {
    color: #854d0e !important;
}

/* Buttons */
.gradio-container button.primary,
.gradio-container button.primary-btn {
    background: #0f766e !important;
    color: #ffffff !important;
    border: 1px solid #0f766e !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.gradio-container button.primary:hover,
.gradio-container button.primary-btn:hover {
    background: #115e59 !important;
    color: #ffffff !important;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    color: #17324d !important;
    border-color: #cbd5e1 !important;
}

/* Image upload and result areas */
.gradio-container .image-container,
.gradio-container .upload-container {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
}

/* Footer */
.footer-note {
    text-align: center;
    color: #475569 !important;
    font-size: 12px;
    padding: 20px 5px;
}

/* Mobile */
@media (max-width: 640px) {
    .main-header {
        padding: 20px;
    }

    .section-heading {
        font-size: 19px;
    }
}
"""


# ==================================================
# 3. PROBABILITY BAR DISPLAY
# ==================================================

def probability_html(probabilities):
    normal = max(
        0.0,
        min(100.0, probabilities["Normal"] * 100)
    )

    tumor = max(
        0.0,
        min(100.0, probabilities["Tumor"] * 100)
    )

    return f"""
    <div class="probability-card">
        <div class="probability-heading">
            <span>Normal</span>
            <span>{normal:.2f}%</span>
        </div>
        <div class="probability-track">
            <div class="probability-fill normal-fill"
                 style="width:{normal:.2f}%">
            </div>
        </div>
    </div>

    <div class="probability-card">
        <div class="probability-heading">
            <span>Tumor</span>
            <span>{tumor:.2f}%</span>
        </div>
        <div class="probability-track">
            <div class="probability-fill tumor-fill"
                 style="width:{tumor:.2f}%">
            </div>
        </div>
    </div>
    """


# ==================================================
# 4. PREDICTION AND GRAD-CAM
# ==================================================

def predict_kidney(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error(
            "Please upload a kidney CT image first."
        )

    progress(0.05, desc="Preparing image...")

    original_image = image.convert("RGB").resize(
        (224, 224)
    )

    original = np.asarray(
        original_image,
        dtype=np.uint8
    )

    x = tf.convert_to_tensor(
        original.astype(np.float32)[None, ...]
    )

    try:
        progress(
            0.20,
            desc="Extracting image features..."
        )

        # Extract features from the trained backbone
        augmented = augmentation(
            x,
            training=False
        )

        feature_maps = backbone(
            augmented,
            training=False
        )

        progress(
            0.55,
            desc="Calculating class probabilities..."
        )

        # Grad-CAM calculation
        with tf.GradientTape() as tape:
            tape.watch(feature_maps)

            pooled = pooling(feature_maps)

            dropped = dropout(
                pooled,
                training=False
            )

            predictions = classifier(dropped)

            class_index = tf.argmax(
                predictions[0],
                output_type=tf.int32
            )

            score = predictions[:, class_index]

        progress(
            0.70,
            desc="Generating Grad-CAM explanation..."
        )

        gradients = tape.gradient(
            score,
            feature_maps
        )

        if gradients is None:
            raise gr.Error(
                "Grad-CAM gradients could not be calculated."
            )

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True
        )

        heatmap = tf.reduce_sum(
            weights * feature_maps,
            axis=-1
        )[0]

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        heatmap = heatmap.numpy()

        # Convert predictions into class probabilities
        probability_values = predictions.numpy()[0]

        probabilities = {
            CLASS_NAMES[i]: float(probability_values[i])
            for i in range(len(CLASS_NAMES))
        }

        predicted_class = CLASS_NAMES[
            int(np.argmax(probability_values))
        ]

        # Create the Grad-CAM overlay
        heatmap_image = Image.fromarray(
            np.uint8(heatmap * 255)
        ).resize((224, 224))

        colored = plt.get_cmap("jet")(
            np.asarray(
                heatmap_image,
                dtype=np.float32
            ) / 255.0
        )[..., :3]

        overlay = (
            0.55 * original.astype(np.float32) / 255.0
            + 0.45 * colored
        )

        overlay = np.uint8(
            np.clip(overlay, 0, 1) * 255
        )

        progress(
            0.90,
            desc="Preparing results..."
        )

        interpretation = (
            f"### Model prediction: {predicted_class}\n\n"
            "This result represents the model's selected class, "
            "not a medical diagnosis. The displayed probabilities "
            "are model outputs and may not be clinically calibrated. "
            "Grad-CAM highlights regions that influenced the model; "
            "it does not confirm tumor boundaries."
        )

        progress(
            1.0,
            desc="Analysis complete"
        )

        return (
            original,
            overlay,
            probability_html(probabilities),
            interpretation,
            "Analysis completed successfully."
        )

    except gr.Error:
        raise

    except Exception as error:
        print(
            f"Prediction error: {error}",
            flush=True
        )

        raise gr.Error(
            "Analysis failed. Please check the Render logs."
        ) from error

    finally:
        gc.collect()


# ==================================================
# 5. WEBSITE INTERFACE
# ==================================================

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="teal",
        secondary_hue="teal",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Kidney XAI | Image Analysis"
) as demo:

    # Header
    gr.HTML("""
    <div class="main-header">
        <div class="eyebrow">AI Research Project</div>
        <h1>Kidney XAI</h1>
        <p>
            An explainable image classification research prototype
            using EfficientNetB0 and Grad-CAM to explore kidney CT
            image classification.
        </p>
        <span class="status-pill">
            EfficientNetB0 &nbsp;·&nbsp; Grad-CAM
        </span>
    </div>
    """)

    # Upload and classification sections
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="section-heading">
                01 · Upload image
            </div>
            <div class="section-subtitle">
                Drag and drop a kidney CT image below, or select
                an image from your device.
            </div>
            """)

            input_image = gr.Image(
                type="pil",
                label="Kidney CT image",
                sources=["upload"],
                height=300
            )

            analyze_button = gr.Button(
                "Analyze image",
                variant="primary"
            )

            clear_button = gr.Button(
                "Clear and start again"
            )

            gr.HTML("""
            <div class="info-box">
                <strong>How it works</strong><br>
                1. Upload an image.<br>
                2. Run the trained classification model.<br>
                3. Review probabilities and Grad-CAM.
            </div>
            """)

        with gr.Column(scale=1):
            gr.HTML("""
            <div class="section-heading">
                02 · Classification results
            </div>
            <div class="section-subtitle">
                Review the model's predicted class and output scores.
            </div>
            """)

            probability_output = gr.HTML(
                value="""
                <div class="panel">
                    <strong>Waiting for an image</strong>
                    <p>
                        Class probabilities will appear here
                        after analysis.
                    </p>
                </div>
                """
            )

            status_output = gr.Markdown(
                "Ready. Upload an image to begin."
            )

    # Explainability
    gr.HTML("""
    <div class="section-heading">
        03 · Explainability
    </div>
    <div class="section-subtitle">
        Compare the original image with its Grad-CAM visualization.
    </div>
    """)

    with gr.Row():
        original_output = gr.Image(
            label="Original image",
            type="numpy",
            interactive=False
        )

        gradcam_output = gr.Image(
            label="Grad-CAM overlay",
            type="numpy",
            interactive=False
        )

    interpretation_output = gr.Markdown(
        "The model interpretation will appear here after analysis."
    )

    # About project
    gr.HTML("""
    <div class="section-heading">
        About this project
    </div>
    <div class="panel">
        <p>
            <strong>Kidney XAI</strong> is an educational research
            prototype that uses a trained EfficientNetB0 model to
            classify supplied kidney CT images into the model's
            Normal and Tumor categories.
        </p>
        <p>
            <strong>Explainability:</strong> Grad-CAM generates a
            heatmap of image regions that influence the model's
            selected prediction.
        </p>
        <p>
            <strong>Technologies:</strong> Python, TensorFlow,
            EfficientNetB0, Grad-CAM and Gradio.
        </p>
    </div>
    """)

    # Limitations and disclaimer
    gr.HTML("""
    <div class="section-heading">
        Important limitations
    </div>
    <div class="warning-box">
        <strong>
            Research use only — not a medical diagnostic tool.
        </strong>
        <br><br>
        Results depend on the training data and model. The displayed
        probabilities are not necessarily calibrated confidence
        estimates. Grad-CAM is an approximate explanation and does
        not confirm tumor location, boundaries, or the presence
        of disease. Do not use this application to make medical
        decisions. Consult a qualified healthcare professional.
    </div>
    """)

    # Footer
    gr.HTML("""
    <div class="footer-note">
        Kidney XAI · Educational Research Prototype ·
        EfficientNetB0 + Grad-CAM
    </div>
    """)

    # Run prediction
    analyze_button.click(
        fn=predict_kidney,
        inputs=[input_image],
        outputs=[
            original_output,
            gradcam_output,
            probability_output,
            interpretation_output,
            status_output
        ],
        concurrency_limit=1
    )

    # Clear all inputs and outputs
    clear_button.click(
        fn=lambda: (
            None,
            None,
            None,
            None,
            "Ready. Upload an image to begin."
        ),
        inputs=[],
        outputs=[
            input_image,
            original_output,
            gradcam_output,
            probability_output,
            interpretation_output,
            status_output
        ],
        concurrency_limit=1
    )


# ==================================================
# 6. LAUNCH APPLICATION ON RENDER
# ==================================================

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", "7860")
        ),
        share=False,
        show_error=True
    )
