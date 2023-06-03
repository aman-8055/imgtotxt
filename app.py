import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

st.title("Image Captioning")

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Get user input
img_url = st.text_input("Enter image URL:")
text = st.text_input("Enter text:")

if img_url:
    try:
        # Load image from URL
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        st.image(raw_image, caption="Input Image", use_column_width=True)

        # Conditional image captioning
        inputs = processor(raw_image, text, return_tensors="pt")

        out = model.generate(**inputs)
        caption_conditional = processor.decode(out[0], skip_special_tokens=True)
        st.subheader("Conditional Image Caption:")
        st.write(caption_conditional)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
        st.subheader("Unconditional Image Caption:")
        st.write(caption_unconditional)
    except Exception as e:
        st.error(f"Error: {e}")
