
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "MODELS",
    "kidney_xai_model_v2.keras"
)

CLASS_NAMES = ["Normal", "Tumor"]
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

augmentation = model.get_layer("data_augmentation")
backbone = model.get_layer("efficientnetb0")
pooling = model.get_layer("global_average_pooling2d")
dropout = model.get_layer("dropout")
classifier = model.get_layer("dense")

def predict_kidney(image):
    if image is None:
        raise gr.Error("Please upload an image first.")

    image = image.convert("RGB").resize((224, 224))
    original = np.asarray(image)
    x = tf.convert_to_tensor(
        original.astype(np.float32)[None, ...]
    )

    # Exact same layer sequence as the saved classifier.
    with tf.GradientTape() as tape:
        augmented = augmentation(x, training=False)
        feature_maps = backbone(augmented, training=False)
        tape.watch(feature_maps)

        pooled = pooling(feature_maps)
        dropped = dropout(pooled, training=False)
        predictions = classifier(dropped)

        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]

    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise gr.Error("Grad-CAM gradients could not be calculated.")

    weights = tf.reduce_mean(
        gradients, axis=(1, 2), keepdims=True
    )
    heatmap = tf.reduce_sum(
        weights * feature_maps, axis=-1
    )[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    probabilities = predictions.numpy()[0]
    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }
    predicted_class = CLASS_NAMES[int(np.argmax(probabilities))]

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize((224, 224))
    colored = plt.get_cmap("jet")(
        np.asarray(heatmap_image) / 255.0
    )[..., :3]

    overlay = (
        0.55 * original / 255.0 + 0.45 * colored
    )
    overlay = np.uint8(np.clip(overlay, 0, 1) * 255)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(original)
    axes[0].set_title("Uploaded image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM overlay")
    axes[1].axis("off")
    fig.tight_layout()

    message = (
        f"Model prediction: {predicted_class}. "
        "Research prototype only; not a medical diagnosis. "
        "Grad-CAM highlights influential regions and does not "
        "confirm tumor boundaries."
    )
    return results, fig, message

demo = gr.Interface(
    fn=predict_kidney,
    inputs=gr.Image(type="pil", label="Upload kidney CT image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Class probabilities"),
        gr.Plot(label="Explainability — Grad-CAM"),
        gr.Textbox(label="Interpretation and limitation")
    ],
    title="Kidney XAI — Image Classification",
    description=(
        "EfficientNetB0 image classification with Grad-CAM. "
        "Educational research prototype; not for clinical use."
    )
)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, show_error=True)
