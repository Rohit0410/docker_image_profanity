from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from flask import jsonify,request,Flask

app = Flask(__name__)

# Load the model and the feature extractor
# model_name = r"D:\Rohit\Image_Profanity\vit-base-profanity-demo-v5-2"
model = AutoModelForImageClassification.from_pretrained("LinkCxO-Abhishekk/VIT_ImageProfane_FineTuned_V2.3")
feature_extractor = AutoFeatureExtractor.from_pretrained("LinkCxO-Abhishekk/VIT_ImageProfane_FineTuned_V2.3")

def predict(image):
    """ Run model prediction on the passed image and return top class """
    # Convert image to RGB, preprocess and add batch dimension
    if image.mode != 'RGB':
        image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    # Retrieve the highest probability class index
    predicted_class_idx = logits.argmax(-1).item()
    result = model.config.id2label[predicted_class_idx]
    data = {'label':result}
    return data

@app.route('/detect', methods=['POST'])
def detect_profanity():
    image = request.files['image']
    if image is not None:
        image = Image.open(image)
        label = predict(image)
    
    return jsonify(label)

if __name__ == '__main__':
    app.run(debug=True)