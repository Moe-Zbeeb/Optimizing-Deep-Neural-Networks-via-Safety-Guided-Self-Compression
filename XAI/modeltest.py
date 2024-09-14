import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define the CNN architecture (SimpleCNN) for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Changed input channels to 1 for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Adjusted to 3 * 3 for MNIST 28x28 size
        self.fc2 = nn.Linear(256, 10)  # 10 classes for MNIST
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to load the model
def load_model(model_class, model_path="/home/mohammad/Desktop/Safety-Driven-Self-Compressing-Neural-Networks/XAI/cnn_mnist.pth"):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to test the model on MNIST test data
def test_model_on_mnist(model_path="/home/mohammad/Desktop/Safety-Driven-Self-Compressing-Neural-Networks/XAI/cnn_mnist.pth"):
    # Load the model
    model = load_model(SimpleCNN, model_path)

    # Define the transformation for MNIST
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load MNIST test set
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    # Iterate through the test set and make predictions
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on MNIST test images: {100 * correct / total:.2f}%")

# Test the function
test_model_on_mnist(model_path="/home/mohammad/Desktop/Safety-Driven-Self-Compressing-Neural-Networks/XAI/cnn_mnist.pth")
