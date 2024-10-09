import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchcam.methods import GradCAM
from sklearn.cluster import KMeans
import numpy as np
import os

# Define SimpleCNN Model Architecture for MNIST (Grayscale)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Adjusted for MNIST's 28x28 size
        self.fc2 = nn.Linear(256, 10)  # 10 classes for MNIST
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)  # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to Load the Model
def load_model(model_class, model_path="/home/mohammad/Desktop/Safety-Driven-Self-Compressing-Neural-Networks/XAI/cnn_mnist.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def compute_gradcam_intensity(images, labels, model, cam_layer='conv3'):
    device = next(model.parameters()).device
    activations = []
    
    with GradCAM(model, target_layer=cam_layer) as cam_extractor:
        model.train()
        for image, label in zip(images, labels):
            image = image.to(device).requires_grad_(True)
            with torch.set_grad_enabled(True):
                output = model(image.unsqueeze(0))
                prediction = output.argmax(dim=1).item()
                cam = cam_extractor(prediction, output)
                intensity = cam[0].sum().item()
                activations.append((image.cpu().detach(), intensity, label))  # Include label
    
    return activations


# Function for Uncertainty Sampling
def get_uncertain_examples(images, labels, model, threshold=0.2):
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    uncertain_examples = []
    uncertain_labels = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        uncertainties = 1 - probabilities.max(dim=1)[0]
        for i, uncertainty in enumerate(uncertainties):
            if uncertainty > threshold:
                uncertain_examples.append(images[i].cpu())
                uncertain_labels.append(labels[i].cpu())
    return uncertain_examples, uncertain_labels

# Function for Clustering
def get_embedding(model, image):
    device = next(model.parameters()).device
    image = image.to(device)
    model.eval()  # Ensure the model is in eval mode
    with torch.no_grad():
        output = model.conv3(model.relu(model.conv2(model.relu(model.conv1(image.unsqueeze(0))))))
        return output.view(output.size(0), -1)

def get_diverse_examples(images, labels, model, num_clusters=10):
    embeddings = []
    images_list = []
    labels_list = []
    for image, label in zip(images, labels):
        embedding = get_embedding(model, image)
        embeddings.append(embedding.squeeze().cpu().numpy())
        images_list.append(image.cpu())
        labels_list.append(label.cpu())
    
    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)))
    clusters = kmeans.fit_predict(embeddings)
    
    selected_images = []
    selected_labels = []
    for cluster in range(num_clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
        if cluster_indices:
            selected_images.append(images_list[cluster_indices[0]])
            selected_labels.append(labels_list[cluster_indices[0]])
    
    return selected_images, selected_labels

# Function to Select Top 1000 Images Using All Techniques
def select_top_examples(testloader, model, num_examples=1000):
    all_activations = []
    all_uncertain = []
    all_diverse = []
    all_labels = []

    for images, labels in testloader:
        # Grad-CAM Intensity
        activations = compute_gradcam_intensity(images, labels, model)
        all_activations.extend(activations)
        
        # Uncertainty Sampling
        uncertain_examples, uncertain_labels = get_uncertain_examples(images, labels, model, threshold=0.2)
        all_uncertain.extend(uncertain_examples)
        all_labels.extend(uncertain_labels)
        
        # Clustering-based Diversity Sampling
        diverse_examples, diverse_labels = get_diverse_examples(images, labels, model)
        all_diverse.extend(diverse_examples)
        all_labels.extend(diverse_labels)
    
    # Sort by Grad-CAM intensity and select top images
    sorted_activations = sorted(all_activations, key=lambda x: x[1], reverse=True)
    top_activations = sorted_activations[:num_examples // 2]
    top_examples_by_gradcam = [img for img, _, _ in top_activations]
    labels_by_gradcam = [label for _, _, label in top_activations]
    
    # Combine examples and labels from all methods
    combined_examples = top_examples_by_gradcam + all_uncertain + all_diverse
    combined_labels = labels_by_gradcam + all_labels
    
    # Remove duplicates while preserving labels
    unique_dict = {}
    for img, label in zip(combined_examples, combined_labels):
        key = tuple(img.numpy().flatten())
        if key not in unique_dict:
            unique_dict[key] = (img, label)
    
    unique_items = list(unique_dict.values())
    unique_images = [item[0] for item in unique_items]
    unique_labels = [item[1] for item in unique_items]
    
    # Truncate if necessary
    if len(unique_images) > num_examples:
        unique_images = unique_images[:num_examples]
        unique_labels = unique_labels[:num_examples]
    
    return unique_images, unique_labels

# Main function to load model, apply techniques, and save selected images with labels
def create_safety_set(model_path, save_path="safety_set_images"):
    # Load the model
    model = load_model(SimpleCNN, model_path)

    # Define the transformation for MNIST
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load MNIST test set
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Select top 1000 critical images and their corresponding labels
    top_1000_images, top_1000_labels = select_top_examples(testloader, model, num_examples=1000)

    # Save images and labels to the specified folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, (img, label) in enumerate(zip(top_1000_images, top_1000_labels)):
        img = img * 0.5 + 0.5  # Unnormalize the image
        img_path = os.path.join(save_path, f"image_{i}_label_{label}.png")  # Save with label in the filename
        torchvision.utils.save_image(img, img_path)

    print(f"Saved {len(top_1000_images)} images to {save_path}")

# Call the function with the path to your saved model
create_safety_set("/home/mohammad/Desktop/Safety-Driven-Self-Compressing-Neural-Networks/XAI/cnn_mnist.pth", "safety_set_images_d")
