# VisioQuest-AI-Image-Inquiry-

title: VisioQuest AI Image Inquiry
emoji: 🏢
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false

## Overview
VisioQuest is an advanced Visual Question Answering (VQA) project that leverages the power of Vision Transformers (ViT) and RoBERTa to answer questions about images. This project utilizes the DAQUAR dataset and integrates a multimodal model to process both textual and image data for accurate question answering.

## Demo
<p float="left">
  <img src="/images/img1.png" width="480" />
  <img src="/images/img2.png" width="480" /> 
  <img src="/images/img3.png" width="480" />
  <img src="/images/img4.png" width="480" />
  <img src="/images/img5.png" width="480" /> 
  <img src="/images/img6.png" width="480" />
</p>

## Dataset
The DAQUAR dataset is a pioneering VQA dataset, consisting of 6794 training and 5674 test question-answer pairs derived from the NYU-Depth V2 Dataset. Each image in the dataset is associated with an average of nine question-answer pairs. The dataset is pre-processed version of the Full DAQUAR Dataset. You can access it from here: https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset?select=test_images_list.txt

## Multimodal Collator for the Dataset
The project includes a custom collator designed to be used within the Trainer() to efficiently construct the DataLoader from the dataset. This collator handles both textual (question) and image data, performing the following tasks:
  1. Tokenizing the question text and generating attention masks.
  2. Featurizing the images by processing their pixel values.
These processed inputs are then fed into the multimodal transformer model, enabling the question-answering functionality.

## Multimodal VQA Model Architecture
The multimodal VQA model in this project is a fusion model that integrates information from text and image encoders to perform the VQA task. The architecture is as follows:

  1. Text Encoder: A transformer model like BERT or RoBERTa processes the tokenized questions.
  2. Image Encoder: An image transformer like ViT processes the image features.
  3. Fusion: The outputs of the text and image encoders are concatenated and passed through a fully-connected network.
  4. Output: The network's output matches the dimensions of the answer space.
The model is trained using Cross-Entropy Loss, appropriate for the multi-class classification nature of the VQA task.

## Performance Metrics
To evaluate the performance of the VQA model, the Wu & Palmer similarity metric is used. This metric measures the semantic similarity between two words or phrases based on their positions in a taxonomy and their relative location to their Least Common Subsumer (LCS).

**Wu & Palmer Similarity:** Effective for single-word answers, which are the primary focus of this task. The Natural Language Toolkit (nltk) provides an implementation of this similarity score using the WordNet taxonomy, aligning with the DAQUAR dataset's requirements.


