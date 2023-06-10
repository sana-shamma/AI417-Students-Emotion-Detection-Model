# import 
import torch
import torch.nn as nn                   #for sequence api in torch
from torch.utils.data import DataLoader #for loading images
import numpy as np                      #just in case if you need numpy arrays
import torchvision.transforms as T      #Used for data preprocessing and converting images to tensors
import torchvision.datasets as dset
import torch.optim as optim             #For using the desired parameter update
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

print("Using device: ",device)

#-------------------------------------------Loading Dataset-----------------------------------------------------#
transform = T.Compose([T.RandomHorizontalFlip(), T.Grayscale(num_output_channels=3), T.ToTensor()])

#Training 
dataset = dset.ImageFolder('AI-417-Project/train',transform=transform)

# Splitting the dataset into train and test sets with 80% for training and 20% for testing
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)

loaded_train = DataLoader(train_data,batch_size=64,shuffle=True)
loaded_validation = DataLoader(validation_data,batch_size=64,shuffle=True)
loaded_test = DataLoader(test_data,batch_size=64,shuffle=True)

loss_history = []
validation_acc = []
training_acc = []
#--------------------------------------------------------------------------------------------------------------#  

#--------------------Creating a method for predicting validation accuracy--------------------------------------#
#     
# Computes the accuracy of the given model on the given data loader.
# Args:
#     loader: A PyTorch DataLoader object that provides a stream of input data.
#     model: A PyTorch model object that takes input data and produces output scores.
# Returns:
#     None. Prints the accuracy of the model on the given data loader. 
# 
def check_accuracy_part(loader, model):
    print('Checking accuracy on validation set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
#---------------------------------------------------------------------------------------------------------------#

        #-----------------------------------Visualizing the image-------------------------------------------------------#           
import torch
import torchvision
import random

# Assume that 'loaded_train' is a PyTorch DataLoader object containing the dataset
dataiter = iter(loaded_train)
images, labels = next(dataiter)

# Create a dictionary for mapping labels to expressions
expression = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral"}

# Select a random image to display
random_idx = random.randint(0, 63)
print("Target label: ", expression[int(labels[random_idx].item())])

# Convert the image tensor to a numpy array and transpose the dimensions to match matplotlib's format
image_np = images[random_idx].permute(1, 2, 0).numpy()

# Convert the data type to uint8
image_np = (image_np * 255).astype(np.uint8)

# Display the image using PyTorch's built-in image display function
torchvision.transforms.functional.to_pil_image(image_np).show()
#--------------------------------------------------------------------------------------------------------#

#----------------------------------Trianing Model--------------------------------------------------------#
#
# Trains the given model on the given optimizer using the given number of epochs.
# Args:
#     model: A PyTorch model object to be trained.
#     optimizer: A PyTorch optimizer object to use for gradient descent.
#     epochs: An integer specifying the number of epochs to train for (default 1).
# Returns:
#     None. Trains the model in-place and prints the loss and accuracy during training.
#
def train_part(model, optimizer, epochs=1):

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print("epoch: ",e+1)
        for t, (x, y) in enumerate(loaded_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 100 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part(loaded_validation, model)
                print()
#--------------------------------------------------------------------------------------------------#

                #-------------------------------------Arh Model----------------------------------------------------#

model = None
optimizer = None

#First architecture #1,32,32
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
model = nn.Sequential(
    conv1,
    conv2,
    conv3,
    conv4,
    fc
)

learning_rate=0.001
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
train_part(model, optimizer, epochs=50)
#--------------------------------------------------------------------------------------------#

# Save the trained model
torch.save(model.state_dict(), 'pretrained_model.pth')
