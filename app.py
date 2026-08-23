import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import json
from google import genai

st.set_page_config(page_title="Solar Defect Inspector", layout="wide")
st.title("☀️ Solar Infrastructure AI Inspector")

@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    elif os.path.exists('/content/runs/detect/train-2/weights/best.pt'):
        return YOLO('/content/runs/detect/train-2/weights/best.pt')
    return YOLO("yolov8n.pt")

model = load_model()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

uploaded_file = st.file_uploader("Upload an inspection image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Survey Image", use_container_width=True)
    
    with st.spinner("Analyzing panel imagery..."):
        results = model(image, conf=0.10)
        with col2:
            st.image(results[0].plot(), caption="Computer Vision Detections", use_container_width=True)
        
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })
            
    st.subheader("📋 Automated Site Manager Inspection Report")
    with st.spinner("Generating plain-language report with Gemini..."):
        if not client:
            st.warning("⚠️ GEMINI_API_KEY is not configured in environment secrets.")
        else:
            if not detections:
                data_str = "No defects detected by the current model."
            else:
                data_str = json.dumps(detections)
                
            prompt = f"Write a professional 3-sentence executive summary of solar panel defects based on this detection data: {data_str}. Keep it plain-language for a site manager."
            
            try:
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")
