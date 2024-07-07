
# Students Emotion Detection Using Convolutional Neural Networks 🧑‍🎓🤖

This project involves building and training a Convolutional Neural Network (CNN) model to classify human emotions from images. It includes Python scripts for training the model and using the trained model for real-time emotion classification from a webcam feed.

## Project Overview 🌐
To achieve UPM's objective of providing excellent academic programs and a supportive learning environment for students, a deep learning-based model for emotion detection using face recognition can be designed and developed. This model can help monitor student engagement and emotional responses to learning materials during lectures. Additionally, it can be used for statistical research to analyze student reactions to various activities and events organized by different entities at UPM (e.g., clubs) to rank them based on their positive influence.

Overall, this project will help the instructors gain insights into the effectiveness of their teaching methods and adjust them as needed. In addition, it offers objective measures of emotional responses, allowing for better-informed decision-making regarding future actions. In the end, this will lead to in a more productive and interesting learning environment for UPM students.

##  Project Structure 🏗️
### Training Script (`Train.py`) 🏋️ 
The training script performs the following tasks:

1. **Loads and Preprocesses the Dataset**: Utilizes torchvision to load images from the 'dataset' directory, applies transformations (random horizontal flip and grayscale conversion), and splits the data into training, validation, and test sets.
2. **Defines the CNN Architecture**: Constructs a CNN with four convolutional layers followed by batch normalization, ReLU activations, max pooling, and a fully connected layer.
3. **Trains the Model**: Implements a training loop that trains the model using the cross-entropy loss function and the Adam optimizer. The script also includes functionality to check validation accuracy during training.
4. **Saves the Trained Model**: After training, the model's state dictionary is saved to a file named `pretrained_model.pth`.

### Testing Script (`Test.py`) 📹
The testing script performs the following tasks:

1. **Loads the Pretrained Model**: Loads the previously trained model from the `pretrained_model.pth` file and prepares it for inference.
2. **Defines the Emotion Labels**: Maps the model's output indices to emotion labels (angry, disgust, fear, happy, neutral).
3. **Classifies Emotions in Real-Time**: Captures frames from the webcam, preprocesses them, and uses the model to predict the emotion. The predicted emotion is displayed on the video feed in real-time.

## Running the Project 🚀

### Load Dataset 📥 
From here: [FER13 Dataset](https://www.kaggle.com/datasets/gauravsharma99/fer13-cleaned-dataset/data)

### Training the Model 🏋️
1. Ensure you have the necessary data in the `dataset` directory.
2. Run the training script:
    ```bash
    python Train.py
    ```
3. The script will train the model and save the trained model to `pretrained_model.pth`.

### Testing the Model 🎥
1. Ensure `pretrained_model.pth` is present in the same directory as `Test.py`.
2. Run the testing script:
    ```bash
    python Test.py
    ```
3. The script will open a webcam feed and display the predicted emotion for each frame.

## Requirements 📋
- Python 3.x 
- PyTorch 
- NumPy 
- OpenCV 
- scikit-learn 
- torchvision 
- matplotlib 

## Directory Structure 📂
```
.
├── Train.py
├── Test.py
├── pretrained_model.pth
└── dataset
    ├── angry 
    ├── disgust 
    ├── fear 
    ├── happy 
    └── neutral 
```

#### Contributors ✍️

- Aisha Ahmad
- Salwa Shama
- Samah Shama
- Sana Shamma

