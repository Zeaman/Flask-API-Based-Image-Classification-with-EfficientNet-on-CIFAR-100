import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from collections import OrderedDict
import torchvision

app = Flask(__name__)

# Define the model architecture
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomEfficientNet, self).__init__()
        # Create an EfficientNet-B0 without pretrained weights
        self.model = models.efficientnet_b0(pretrained=False)
        # Replace the classifier to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize model
model = CustomEfficientNet(num_classes=100)

# Load model weights
state_dict = torch.load("efficientnet_cifar100.pth", map_location=torch.device('cpu'))
# Adjust the keys by adding the "model." prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = "model." + k  # add the prefix expected by our model
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

# Image transformation (adjust input size as needed; here we use 224x224 as used in EfficientNet-B0)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-100 class names
cifar100_classes = torchvision.datasets.CIFAR100(root="./data", train=True, download=True).classes
class_names = cifar100_classes

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if file is provided
        if "file" not in request.files:
            return render_template("index.html", message="No file part", image_path=None)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file", image_path=None)

        # Save image with a unique filename to avoid overwrites
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        counter = 1
        while os.path.exists(save_path):
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(UPLOAD_FOLDER, f"{name}_{counter}{ext}")
            counter += 1
        
        file.save(save_path)
        return render_template("index.html", image_path=save_path)

    return render_template("index.html", image_path=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the path of the uploaded image from the form data
    image_path = request.form.get("image_path")
    if not image_path:
        return render_template("index.html", message="Please upload an image first", image_path=None)

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()  # Get the class index
        class_name = class_names[predicted_class]  # Map the class index to the class name

    return render_template("index.html", image_path=image_path, prediction=f"Predicted Class: {class_name}")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
