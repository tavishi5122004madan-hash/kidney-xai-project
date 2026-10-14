
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLBACKEND"] = "Agg"

import gc
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

tf.config.set_visible_devices([], "GPU")

# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "MODELS",
    "kidney_xai_model_v2.keras"
)

CLASS_NAMES = ["Normal", "Tumor"]

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

augmentation = model.get_layer("data_augmentation")
backbone = model.get_layer("efficientnetb0")
pooling = model.get_layer("global_average_pooling2d")
dropout = model.get_layer("dropout")
classifier = model.get_layer("dense")


# --------------------------------------------------
# 2. CUSTOM CLINICAL WHITE AND TEAL DESIGN
# --------------------------------------------------

CSS = """
:root {
    --teal: #0f766e;
    --teal-light: #ccfbf1;
    --navy: #17324d;
    --muted: #64748b;
    --border: #dbe5e8;
    --surface: #ffffff;
    --background: #f4f8fa;
}

.gradio-container {
    background: var(--background) !important;
    font-family: Arial, sans-serif !important;
}

.main-header {
    background: linear-gradient(120deg, #ffffff, #e8f8f5);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 18px;
}

.eyebrow {
    color: var(--teal);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.main-header h1 {
    color: var(--navy);
    font-size: clamp(27px, 4vw, 40px);
    margin: 10px 0;
}

.main-header p {
    color: var(--muted);
    line-height: 1.7;
    max-width: 760px;
}

.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    background: var(--teal-light);
    color: #115e59;
    font-size: 12px;
    font-weight: 700;
}

.section-heading {
    color: var(--navy);
    font-size: 21px;
    font-weight: 750;
    margin: 10px 0;
}

.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
}

.primary-btn {
    background: #0f766e !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.probability-card {
    background: white;
    border: 1px solid #dbe5e8;
    border-radius: 14px;
    padding: 17px;
    margin: 10px 0;
}

.probability-heading {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: #17324d;
    font-weight: 700;
    margin-bottom: 10px;
}

.probability-track {
    height: 12px;
    border-radius: 20px;
    background: #e8eef0;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 20px;
}

.normal-fill {
    background: #0d9488;
}

.tumor-fill {
    background: #d97757;
}

.info-box {
    background: #ffffff;
    border: 1px solid #dbe5e8;
    border-left: 4px solid #0f766e;
    border-radius: 12px;
    padding: 16px;
    color: #475569;
    line-height: 1.7;
}

.warning-box {
    background: #fff8e8;
    border: 1px solid #f3dfad;
    border-radius: 12px;
    padding: 16px;
    color: #854d0e;
    line-height: 1.65;
}

.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 18px 5px;
}

button.primary-btn:hover {
    background: #115e59 !important;
}

@media (max-width: 640px) {
    .main-header {
        padding: 20px;
    }
}
"""


# --------------------------------------------------
# 3. PROBABILITY BAR HTML
# --------------------------------------------------

def probability_html(probabilities):
    normal = max(
        0.0, min(100.0, probabilities["Normal"] * 100)
    )
    tumor = max(
        0.0, min(100.0, probabilities["Tumor"] * 100)
    )

    return f"""
    <div class="probability-card">
        <div class="probability-heading">
            <span>Normal</span>
            <span>{normal:.2f}%</span>
        </div>
        <div class="probability-track">
            <div class="probability-fill normal-fill"
                 style="width:{normal:.2f}%"></div>
        </div>
    </div>

    <div class="probability-card">
        <div class="probability-heading">
            <span>Tumor</span>
            <span>{tumor:.2f}%</span>
        </div>
        <div class="probability-track">
            <div class="probability-fill tumor-fill"
                 style="width:{tumor:.2f}%"></div>
        </div>
    </div>
    """


# --------------------------------------------------
# 4. PREDICTION AND GRAD-CAM
# --------------------------------------------------

def predict_kidney(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload a kidney CT image first.")

    progress(0.05, desc="Preparing image...")

    original_image = image.convert("RGB").resize((224, 224))
    original = np.asarray(original_image, dtype=np.uint8)

    x = tf.convert_to_tensor(
        original.astype(np.float32)[None, ...]
    )

    fig = None

    try:
        progress(0.20, desc="Extracting image features...")

        # Feature extraction
        augmented = augmentation(x, training=False)
        feature_maps = backbone(augmented, training=False)

        progress(0.55, desc="Calculating class probabilities...")

        # Grad-CAM gradient calculation
        with tf.GradientTape() as tape:
            tape.watch(feature_maps)

            pooled = pooling(feature_maps)
            dropped = dropout(pooled, training=False)
            predictions = classifier(dropped)

            class_index = tf.argmax(
                predictions[0],
                output_type=tf.int32
            )
            score = predictions[:, class_index]

        progress(0.70, desc="Generating Grad-CAM explanation...")

        gradients = tape.gradient(score, feature_maps)

        if gradients is None:
            raise gr.Error(
                "Grad-CAM could not calculate image gradients."
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

        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        heatmap = heatmap.numpy()

        # Class probabilities
        probabilities_array = predictions.numpy()[0]

        probabilities = {
            CLASS_NAMES[i]: float(probabilities_array[i])
            for i in range(len(CLASS_NAMES))
        }

        predicted_class = CLASS_NAMES[
            int(np.argmax(probabilities_array))
        ]

        # Resize the heatmap and create overlay
        heatmap_image = Image.fromarray(
            np.uint8(heatmap * 255)
        ).resize((224, 224))

        # Use Pillow to colorize the heatmap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colored = plt.get_cmap("jet")(
            np.asarray(heatmap_image, dtype=np.float32) / 255.0
        )[..., :3]

        overlay = (
            0.55 * original.astype(np.float32) / 255.0
            + 0.45 * colored
        )

        overlay = np.uint8(
            np.clip(overlay, 0, 1) * 255
        )

        progress(0.90, desc="Preparing results...")

        message = (
            f"### Model prediction: {predicted_class}\n\n"
            "This is the model's predicted class, not a medical diagnosis. "
            "The probability values describe the model's outputs and "
            "are not necessarily clinically calibrated. Grad-CAM shows "
            "regions that influenced the prediction; it does not "
            "establish tumor boundaries."
        )

        progress(1.0, desc="Analysis complete")

        return (
            original,
            overlay,
            probability_html(probabilities),
            message,
            "Analysis completed successfully."
        )

    except gr.Error:
        raise

    except Exception as error:
        print(f"Prediction error: {error}", flush=True)
        raise gr.Error(
            "Analysis failed. Please check the Render logs."
        ) from error

    finally:
        gc.collect()


# --------------------------------------------------
# 5. WEBSITE INTERFACE
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Kidney XAI | Image Analysis"
) as demo:

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

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="section-heading">01 · Upload image</div>
            <div class="section-subtitle">
                Drag and drop a kidney CT image below, or select
                one from your device.
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
                variant="primary",
                elem_classes=["primary-btn"]
            )

            clear_button = gr.Button("Clear and start again")

            gr.HTML("""
            <div class="info-box">
                <strong>How it works</strong><br>
                1. Upload an image.<br>
                2. Run the trained classification model.<br>
                3. Review class probabilities and Grad-CAM.
            </div>
            """)

        with gr.Column(scale=1):
            gr.HTML("""
            <div class="section-heading">02 · Classification results</div>
            <div class="section-subtitle">
                Review the predicted class and model output scores.
            </div>
            """)

            probability_output = gr.HTML(
                value="""
                <div class="panel">
                    <strong>Waiting for an image</strong>
                    <p>Class probabilities will appear here after analysis.</p>
                </div>
                """
            )

            status_output = gr.Markdown(
                "Ready. Upload an image to begin."
            )

    gr.HTML("""
    <div class="section-heading">03 · Explainability</div>
    <div class="section-subtitle">
        Compare the uploaded image with the Grad-CAM visualization.
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

    gr.HTML("""
    <div class="section-heading">About this project</div>
    <div class="panel">
        <p>
            <strong>Kidney XAI</strong> is an educational research
            prototype that uses a trained EfficientNetB0 model to
            classify supplied kidney CT images into the model's
            Normal and Tumor categories.
        </p>
        <p>
            <strong>Explainability:</strong> Grad-CAM produces a
            heatmap showing image regions that influence the model's
            selected prediction.
        </p>
        <p>
            <strong>Technology:</strong> Python, TensorFlow,
            EfficientNetB0, Grad-CAM and Gradio.
        </p>
    </div>
    """)

    gr.HTML("""
    <div class="section-heading">Important limitations</div>
    <div class="warning-box">
        <strong>Research use only — not a medical diagnostic tool.</strong>
        <br><br>
        Results depend on the training data and model. The displayed
        probabilities are not necessarily calibrated confidence
        estimates. Grad-CAM is an approximate explanation and does
        not confirm tumor location, boundaries, or the presence
        of disease. Do not use this application to make medical
        decisions. Consult a qualified healthcare professional.
    </div>
    """)

    gr.HTML("""
    <div class="footer-note">
        Kidney XAI · Educational Research Prototype ·
        EfficientNetB0 + Grad-CAM
    </div>
    """)

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

    clear_button.click(
        fn=lambda: (
            None,
            None,
            None,
            None,
            "Ready. Upload an image to begin."
        ),
        outputs=[
            input_image,
            original_output,
            gradcam_output,
            probability_output,
            interpretation_output,
            status_output
        ],
        inputs=[],
        concurrency_limit=1
    )


if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        show_error=True
    )
