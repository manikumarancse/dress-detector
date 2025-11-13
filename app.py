import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import zipfile
import io
import requests

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.strip().split("\n")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_dress_type(image, model, labels):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
    return labels[predicted.item()]

def segment_dress(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            cropped = image[y:y+h, x:x+w]
            segments.append(cropped)
    return segments

def create_zip(segments):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i, seg in enumerate(segments):
            rgb_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_seg)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zipf.writestr(f"segment_{i+1}.png", img_bytes.getvalue())
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="Dress Type & Stitch Splitter", page_icon="👗", layout="wide")
    st.title("👗 Dress Type & Stitch Splitter")
    st.write("Upload an image to detect the dress type and view approximate stitched segments.")

    uploaded_file = st.file_uploader("Upload a dress image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            model = load_model()
            labels = load_labels()
            label = predict_dress_type(image, model, labels)

        st.subheader("Predicted Dress Type")
        st.success(label.capitalize())

        st.subheader("Detected Segments (Approximate)")
        segments = segment_dress(tmp_path)

        if len(segments) == 0:
            st.warning("No significant stitched segments detected.")
        else:
            cols = st.columns(3)
            for i, seg in enumerate(segments):
                rgb_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
                cols[i % 3].image(rgb_seg, caption=f"Segment {i+1}")

            zip_buffer = create_zip(segments)
            st.download_button(
                label="📦 Download All Segments as ZIP",
                data=zip_buffer,
                file_name="dress_segments.zip",
                mime="application/zip"
            )

        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
