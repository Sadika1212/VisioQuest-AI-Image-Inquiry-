import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch.nn as nn
from transformers import (RobertaTokenizer, ViTImageProcessor,
    RobertaModel, AutoConfig, ViTModel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import safetensors


with open("answer_space.txt") as f:
    answer_space = f.read().splitlines()

@dataclass
class MultimodalCollator:
    tokenizer: RobertaTokenizer
    preprocessor: ViTImageProcessor
    
    def tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }
    
    def preprocess_images(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        processed_images = self.preprocessor(images, return_tensors="pt")
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
    def __call__(self, raw_batch_dict) -> Dict[str, torch.Tensor]:
        question_batch = raw_batch_dict['question'] if isinstance(raw_batch_dict, dict) else [i['question'] for i in raw_batch_dict]
        image_id_batch = raw_batch_dict['image'] if isinstance(raw_batch_dict, dict) else [i['image'] for i in raw_batch_dict]
        return {
            **self.tokenize_text(question_batch),
            **self.preprocess_images(image_id_batch),
        }



class MultimodalVQAModel(nn.Module):
    def __init__(self,
        # num_labels: int = len(answer_space),
        num_labels: int = 582,intermediate_dim: int = 512,
        pretrained_text_name: str = 'roberta-base',
        pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):
        
        super(MultimodalVQAModel, self).__init__()
        # Text and image encoders
        
        self.text_encoder = RobertaModel.from_pretrained(pretrained_text_name)
        self.image_encoder = ViTModel.from_pretrained(pretrained_image_name)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size),
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(intermediate_dim, num_labels))
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None):

        # Encode text with masking
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        
        # Encode images
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True)
        
        # Make predictions
        logits = self.fusion(torch.cat([encoded_text['pooler_output'], encoded_image['pooler_output']], dim=1))
        out = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        return out


def create_multimodal_vqa_collator_and_model(text_encoder='roberta-base', image_encoder='google/vit-base-patch16-224-in21k'):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    preprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    multimodal_collator = MultimodalCollator(tokenizer=tokenizer, preprocessor=preprocessor)
    multimodal_model = MultimodalVQAModel().to(device)
    return multimodal_collator, multimodal_model


collator, model= create_multimodal_vqa_collator_and_model()
model.load_state_dict(safetensors.torch.load_file('model.safetensors'))
tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 

# Function to preprocess the image for the model
def preprocess_image(image, image_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(device)

# Function to tokenize the question
def tokenize_question(question,tokenizer):
    return tokenizer(question, return_tensors="pt", padding=True, truncation=True)


# Custom CSS to reduce space between image and text
st.markdown("""
    <style>
    .close-columns .css-1lcbmhc { width: 55% !important; }
    .close-columns .css-1aumxhk { width: 40% !important; margin-left: 5px !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for managing selections
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "predicted_answer" not in st.session_state:
    st.session_state.predicted_answer = None

# Function to reset the state
def reset_state():
    st.session_state.selected_image = None
    st.session_state.question = ""
    st.session_state.predicted_answer = None

# Function to handle image selection
def handle_image_selection(image):
    reset_state()
    st.session_state.selected_image = image.convert("RGB")

# Create the Streamlit application
st.title("VisioQuest: AI-Image-Inquiry")
st.write("The model is trained on the DAQUAR dataset, featuring indoor scenes from the NYU_Depth dataset, including various environments like living rooms, kitchens, bedrooms, bathrooms, grocery stores, and malls. Please upload relevant images for optimal results.")

# Create columns for side-by-side layout
col1, col2 = st.columns(2)

# File uploader
with col1:
    st.subheader("Upload an image:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        handle_image_selection(Image.open(uploaded_file))

# Predefined images gallery
predefined_images = {
    "Image 1": "Test Images/image181.png",
    "Image 2": "Test Images/image126.png",
    "Image 3": "Test Images/image870.png",
    "Image 4": "Test Images/image516.png",
    "Image 5": "Test Images/image1292.png",
    "Image 6": "Test Images/image833.png"
}

# Image selection section
with col2:
    st.subheader("Or drag a test image from here:")
    image_keys = list(predefined_images.keys())

    for i in range(0, len(image_keys), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            index = i + j
            if index < len(image_keys):
                key = image_keys[index]
                image_path = predefined_images[key]
                image = Image.open(image_path)
                col.image(image, caption=None, use_column_width=True)

# Display the selected image
selected_image = st.session_state.selected_image
if selected_image:
    col_image, col_text = st.columns([55, 40], gap="small")
    with col_image:
        st.image(selected_image, width=350)
    with col_text:
        st.write(" ")
        st.write("  \nYou might like to ask: ")
        st.write("-what is in the third rack of the shelf?  \n-what is the object close to the wall     above the stove burner?  \n-what is next to the paper towels?  \n-what color are the bed sheets?  \n-what object is on the ceiling?  \n-what is on the left side of the sink?")
else:
    st.write("Please upload an image or select one from the gallery.")

# Question input section
st.subheader("Ask a question about the image:")
question = st.text_input("Question:", value="", key="question_input")

if st.button("Get Answer"):
    if selected_image and question:
        preprocessed_image = preprocess_image(selected_image)
        question_tokens = tokenize_question(question, tokenizer)

        input_data = {
            "input_ids": question_tokens["input_ids"].to(device),
            "attention_mask": question_tokens["attention_mask"].to(device),
            "pixel_values": preprocessed_image
        }

        model.eval()
        with torch.no_grad():
            output = model(**input_data)
        
        predictions = output["logits"].argmax(axis=-1).cpu().numpy()
        predicted_answer = answer_space[predictions[0]]
        st.session_state.predicted_answer = predicted_answer
        st.write("Predicted Answer:", predicted_answer)
    elif not selected_image:
        st.write("Please upload an image or select one from the predefined set.")
    else:
        st.write("Please enter a question.")




# streamlit run app.py --server.enableXsrfProtection false
        