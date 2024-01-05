import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from torchvision import models
from simple_cnn import SimpleCNN


# Load the pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    st.title("Image Classification using PyTorch and Streamlit")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Prediction
        prediction = predict_image(image)
        
        st.write("")
        st.write("Class Label:")
        st.write(prediction)

if __name__ == '__main__':
    main()
