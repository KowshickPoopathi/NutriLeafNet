"""NutriLeafNet - Streamlit app.
Run:  streamlit run app.py
Expects the folder `best_model/` (saved by the training notebook) next to this file,
or set the env var NUTRILEAF_MODEL_DIR to its location.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification

MODEL_DIR = os.environ.get("NUTRILEAF_MODEL_DIR", "best_model")
IMG_SIZE = 224
LOW_CONF = 0.60          # below this we warn the user
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),    # same as training
])


@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(MODEL_DIR, attn_implementation="eager")
    return model.to(device).eval()


@torch.no_grad()
def predict_with_attention(model, img):
    """Return class probabilities and an attention-rollout heatmap in [0, 1]."""
    x = tfm(img).unsqueeze(0).to(device)
    out = model(pixel_values=x, output_attentions=True)
    probs = out.logits.softmax(-1)[0].cpu().numpy()

    attn = torch.stack(out.attentions).mean(dim=2)                 # average heads
    attn = attn + torch.eye(attn.size(-1), device=attn.device)     # residual connection
    attn = attn / attn.sum(-1, keepdim=True)
    rollout = attn[0]
    for layer in attn[1:]:
        rollout = layer @ rollout
    mask = rollout[0, 0, 1:]
    g = int(mask.numel() ** 0.5)
    mask = mask.reshape(1, 1, g, g)
    mask = F.interpolate(mask, size=(IMG_SIZE, IMG_SIZE), mode="bicubic", align_corners=False)[0, 0]
    mask = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8)).clamp(0, 1).cpu().numpy()
    return probs, mask


st.set_page_config(page_title="NutriLeafNet", page_icon="🌿", layout="wide")
st.title("🌿 NutriLeafNet")
st.caption("Vision Transformer based detection of nutrient deficiencies in cassava leaves")

if not os.path.isdir(MODEL_DIR):
    st.error(f"Model folder '{MODEL_DIR}' not found. Copy the trained `best_model/` folder next to app.py.")
    st.stop()

model = load_model()
class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

file = st.file_uploader("Upload a photo of a cassava leaf", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    probs, mask = predict_with_attention(model, img)
    top = int(np.argmax(probs))

    left, mid, right = st.columns(3)
    with left:
        st.subheader("Uploaded photo")
        st.image(img, use_container_width=True)
    with mid:
        st.subheader("Model attention")
        base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        blend = np.clip(0.55 * base + 0.45 * plt.cm.jet(mask)[..., :3], 0, 1)
        st.image(blend, use_container_width=True, caption="Red = regions the transformer focused on")
    with right:
        st.subheader("Prediction")
        st.metric("Class", class_names[top].replace("_", " ").title())
        st.metric("Confidence", f"{probs[top] * 100:.1f}%")
        if probs[top] < LOW_CONF:
            st.warning("Low confidence. Retake the photo in better light, with the leaf filling the frame.")
        st.bar_chart({c: float(p) for c, p in zip(class_names, probs)})
else:
    st.info("Upload a leaf photo to get a prediction.")
