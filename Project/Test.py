import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
import random
import cv2
import matplotlib.pyplot as plt

# Load the pretrained model state dictionary
conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512 * 3 * 3, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(84,5),
)

# Create an instance of the model

model = nn.Sequential(
    conv1,
    conv2,
    conv3,
    conv4,
    fc
)

model.load_state_dict(torch.load('pretrained_model.pth'))

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

print("Using device:", device)

def check_accuracy_test(x, y, model):
    model.eval()
    with torch.no_grad():
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, preds = scores.max(1)
        return preds

expression = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral"}


transform = T.Compose([T.ToTensor()])

# Load the pre-trained model
model = model
model.to(device=device)
model.eval()

# Function to preprocess and classify a single image frame
def classify_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color channels from BGR to RGB
    frame = cv2.resize(frame, (48, 48))  # Resize the frame to match the input size of the model
    image = transform(frame).unsqueeze(0).to(device)  # Convert frame to tensor and add batch dimension
    predicted_idx = check_accuracy_test(image, torch.tensor([0]), model)  # Provide a dummy label for check_accuracy_test
    return expression[predicted_idx.item()]

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    predicted_label = classify_frame(frame, model)

    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
