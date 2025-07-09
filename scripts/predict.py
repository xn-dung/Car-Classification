# START OF FILE predict.py
import pillow_avif
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import json
import io
import os

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# --- Updated Model Path ---
MODEL_PATH = "C:/Users/nguye/Desktop/School Subjects/2024-2025/2nd semester/Introduction to Artificial Intelligence/Coding/giaodienAI/model/resnet50.pth"  # Use forward slashes

TRAIN_DATA_PATH = os.path.join(project_root, "data", "train")
EXPECTED_CLASSES = 196  # Update as needed

# --- Model Definition ---
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=EXPECTED_CLASSES):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# --- Helper Functions ---
def get_class_names(train_data_path):
    if not os.path.isdir(train_data_path):
        print(f"Error: Train data directory not found at '{train_data_path}'", file=sys.stderr)
        sys.exit(1)
    try:
        class_names = sorted([d for d in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, d))])
        if not class_names:
            print(f"Error: No class subdirectories found in '{train_data_path}'", file=sys.stderr)
            sys.exit(1)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}", file=sys.stderr)
        sys.exit(1)

def load_model(model_path, num_classes):
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)
    try:
        model = ResNet50Model(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from {model_path}", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

def predict_image(image_path, model, class_names):
    try:
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        predicted_class = class_names[predicted_idx]
        probability_dict = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}

        return predicted_class, probability_dict
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main Execution ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Received image path: {image_path}", file=sys.stderr)

    try:
        class_names = get_class_names(TRAIN_DATA_PATH)
        num_classes_found = len(class_names)
        print(f"Found {num_classes_found} classes.", file=sys.stderr)

        if num_classes_found != EXPECTED_CLASSES:
            print(f"Error: Found {num_classes_found} classes, expected {EXPECTED_CLASSES}", file=sys.stderr)
            # sys.exit(1)

        model = load_model(MODEL_PATH, num_classes=EXPECTED_CLASSES)

        model_output_features = model.backbone.fc.out_features
        if model_output_features != EXPECTED_CLASSES:
            print(f"Error: Model output {model_output_features}, expected {EXPECTED_CLASSES}", file=sys.stderr)
            sys.exit(1)

        prediction, probabilities = predict_image(image_path, model, class_names)

        result = {
            "prediction": prediction,
            "probabilities": probabilities
        }
        print(json.dumps(result))

    except SystemExit:
        pass
    except Exception as e:
        error_result = {"error": f"An unexpected error occurred: {e}"}
        print(json.dumps(error_result), file=sys.stdout)
        print(f"Traceback: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
# END OF FILE predict.py
