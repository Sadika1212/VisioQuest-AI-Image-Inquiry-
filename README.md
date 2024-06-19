# VisioQuest-AI-Image-Inquiry-

## Overview
VisioQuest is an advanced Visual Question Answering (VQA) project that leverages the power of Vision Transformers (ViT) and RoBERTa to answer questions about images. This project utilizes the DAQUAR dataset and integrates a multimodal model to process both textual and image data for accurate question answering.

## Demo
<p float="left">
  <img src="/![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/332ddd3c-a134-48bd-9452-87470cedfbcb)" width="100" />
  <img src="/img2.png" width="100" /> 
  <img src="/img3.png" width="100" />
</p>

![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/332ddd3c-a134-48bd-9452-87470cedfbcb) ![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/cd6276b8-5cab-4973-b823-f1ae6f310423)  ![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/f01c6e0a-fee6-4622-accc-3773150533f1)  ![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/3601e4af-760d-4aa2-8782-d0e9fb82aa11)  ![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/60c0afbe-4ed2-47f1-9c03-9eb89df02520)  ![image](https://github.com/Sadika1212/VisioQuest-AI-Image-Inquiry-/assets/57654473/2d77c18c-5c4c-4c3d-9f30-d9946add4c16)

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


