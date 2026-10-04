import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import json
from google import genai

st.set_page_config(page_title="Solar Defect Inspector", layout="wide")
import streamlit as st

# Replace the old heading with the new one
st.title("Solar Panels Infra-Inspection")

# Add the explanation text directly below it
st.markdown("""
### How It Is Used
1. **Take a Picture:** A drone flies over the solar panels and takes a clear photo from above.
2. **Upload the Photo:** You upload that picture directly into the app.
3. **Let the AI Work:** The app acts like a smart assistant, scanning the picture to instantly find any damaged areas.

### Why It Is Used
* **Saves Time:** Instead of a person walking around to check every single solar panel by hand, the system scans them all automatically. 
* **Catches Problems Early:** It spots cracks, hot spots, or dirt before they turn into bigger, more expensive issues.
* **Keeps Power Flowing:** By finding broken panels quickly, you can fix them and make sure the solar farm generates as much electricity as possible.

### What Is the Outcome
* **Highlighted Image:** You get your picture back, but with boxes drawn around the exact spots where the problems are.
* **Simple Health Check:** A list explaining what kind of damage was found and how serious it is.
* **Repair Plan:** Clear advice on which panels need to be cleaned or fixed first, so the maintenance team knows exactly where to go.
""")

# Your existing file upload code will go right below this...

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
