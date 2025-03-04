from flask import Flask, request, jsonify
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Define the CNN model (same structure as before)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# Function to preprocess image
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Define the API route '/menna' to predict digits
@app.route('/menna', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    image = transform_image(file.read())

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    return jsonify({"prediction": prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
