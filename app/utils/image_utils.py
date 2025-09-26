from PIL import Image
import torchvision.transforms as transforms
import io

# Same normalization as used during training
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes to PIL Image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def preprocess_image(image: Image.Image):
    """
    Apply necessary preprocessing to the image.
    Returns a tensor suitable for model input.
    """
    return image_transform(image).unsqueeze(0)  # Add batch dimension
