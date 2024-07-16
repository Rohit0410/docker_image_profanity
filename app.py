# -*- coding: utf-8 -*-
import os
# Set the environment variable to allow duplicate OpenMP libraries (temporary workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
# import cv2
import torch
import faiss
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load a pre-trained model for feature extraction (ResNet)
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# def extract_features(image):
#     if isinstance(image, str):
#         image = cv2.imread(image)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     elif isinstance(image, np.ndarray):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     elif not isinstance(image, Image.Image):
#         image = Image.fromarray(image)

def extract_features(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.convert("RGB"))
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    # image = Image.fromarray(image)  # Convert to PIL Image
    image = transform(image)  # Apply transform to PIL Image
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

# Directory containing image folders
image_base_path = '/usr/src/app/Recent'
# Load the index from disk
index = faiss.read_index("/usr/src/app/image_index_recent.faiss")
image_classes = np.load("/usr/src/app/image_paths_recent.npy").tolist()
image_paths = np.load("/usr/src/app/image_paths_recent.npy").tolist()
class_labels = {'Addiction': 0, 'Harassment,Exploitation and Bullying': 1, 'Hentai_Nudity': 2, 'Neutral': 3, 'Religious_Political': 4, 'Sports and Entertainment': 5, 'Suicides_Accidents': 6, 'Terrorism': 7}

# Function to search for similar images and assign class
def search_and_classify(query_image_path, threshold=0.8, k=5):
    query_features = extract_features(query_image_path)
    query_features = query_features.reshape(1, -1)
    faiss.normalize_L2(query_features)  # Normalize query vector to unit length
    distances, indices = index.search(query_features, k)

    # Ensure indices are within the valid range
    valid_indices = [i for i in indices[0] if i < len(image_paths)]
    # similar_images = [image_paths[i] for i in valid_indices]
    similar_classes = [image_classes[i] for i in valid_indices]

    if len(distances[0]) > 0 and distances[0][0] >= threshold:
        assigned_class = similar_classes[0]
        predicted_class = list(class_labels.keys())[list(class_labels.values()).index(assigned_class)]
        similarity_score = f"{distances[0][0]:.2f}"
        logging.debug(f"Assigned class: {predicted_class} with similarity score:", similarity_score)
        if predicted_class in ['Addiction', 'Harassment,Exploitation and Bullying', 'Hentai_Nudity', 'Religious_Political', 'Suicides_Accidents', 'Terrorism']:
            result = [{"From":"FAISS Vector Database"}, {"label":predicted_class}, {"score":similarity_score}]
        else:
            result = [{"From":"FAISS Vector Database"}, {"label":predicted_class}, {"score":similarity_score}]
        return [str(item) for item in result]
    else:
        logging.debug('It is passed into LLM model')
        model_path = r"AkashSKulkarni/ImageProfanity"
        model = ViTForImageClassification.from_pretrained(model_path)
        image_processor = ViTImageProcessor.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        class_names = ['Addiction','Harassment,Exploitation and Bullying', 'Hentai_Nudity', 'Neutral', 'Religious_Political','Sports and Entertainment','Suicides_Accidents','Terrorism']

        def predict_image(image_array):
            image = image_array.convert("RGB")  
            # image = Image.fromarray(image_array)  # Convert to PIL Image
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_class_idx = torch.max(logits, 1)
            return predicted_class_idx.item(), probabilities

        query_image = Image.open(query_image_path)
        predicted_class_idx, probabilities = predict_image(query_image)
        predicted_class = class_names[predicted_class_idx]
        probability_score = f"{probabilities[0][predicted_class_idx].item():.2f}"
        logging.debug(f'Assigned Class: {predicted_class} with probability score: {probability_score}')
        if predicted_class in ['Addiction', 'Harassment,Exploitation and Bullying', 'Hentai_Nudity', 'Religious_Political', 'Suicides_Accidents', 'Terrorism']:
            result = [{"From":"LLM Model"}, {"label":predicted_class}, {"score":probability_score}]
        else:
            result = [{"From":"LLM Model"}, {"label":predicted_class}, {"score":probability_score}]

        return [str(item) for item in result]

@app.route('/detect', methods=['POST'])
def detect_profanity():
    Folder_path = r"/usr/src/app/image_temp"
    logging.debug('Received request')
    if 'image' not in request.files:
        logging.error('No image file provided')
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    logging.debug(f'Image file received: {image_file.filename}')
    image_path = os.path.join(Folder_path, image_file.filename)
    image_file.save(image_path)
    logging.debug(f'Image file saved at: {image_path}')

    # Read the saved image from disk
    img = Image.open(image_path)

    prediction = search_and_classify(image_path)  # Pass the image path instead of the image itself

    # Convert prediction to a list of strings
    result = [str(item) for item in prediction]

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,port=5002)
