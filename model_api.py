import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Đảm bảo các thư viện AI được import trước FastAPI/uvicorn nếu có vấn đề về thứ tự
import pillow_avif
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
# Xác định thư mục gốc của dự án dựa trên vị trí file model_api.py
PROJECT_ROOT = Path(__file__).resolve().parent

# --- QUAN TRỌNG: Sử dụng đường dẫn tương đối ---
MODEL_PATH: Path = PROJECT_ROOT / "model" / "resnet50.pth"

EXPECTED_CLASSES: int = 196

# --- Class Names ---
CLASS_NAMES: List[str] = [
    'AM General Hummer SUV 2000',
    'Acura Integra Type R 2001',
    'Acura RL Sedan 2012',
    'Acura TL Sedan 2012',
    'Acura TL Type-S 2008',
    'Acura TSX Sedan 2012',
    'Acura ZDX Hatchback 2012',
    'Aston Martin V8 Vantage Convertible 2012',
    'Aston Martin V8 Vantage Coupe 2012',
    'Aston Martin Virage Convertible 2012',
    'Aston Martin Virage Coupe 2012',
    'Audi 100 Sedan 1994',
    'Audi 100 Wagon 1994',
    'Audi A5 Coupe 2012',
    'Audi R8 Coupe 2012',
    'Audi RS 4 Convertible 2008',
    'Audi S4 Sedan 2007',
    'Audi S4 Sedan 2012',
    'Audi S5 Convertible 2012',
    'Audi S5 Coupe 2012',
    'Audi S6 Sedan 2011',
    'Audi TT Hatchback 2011',
    'Audi TT RS Coupe 2012',
    'Audi TTS Coupe 2012',
    'Audi V8 Sedan 1994',
    'BMW 1 Series Convertible 2012',
    'BMW 1 Series Coupe 2012',
    'BMW 3 Series Sedan 2012',
    'BMW 3 Series Wagon 2012',
    'BMW 6 Series Convertible 2007',
    'BMW ActiveHybrid 5 Sedan 2012',
    'BMW M3 Coupe 2012',
    'BMW M5 Sedan 2010',
    'BMW M6 Convertible 2010',
    'BMW X3 SUV 2012',
    'BMW X5 SUV 2007',
    'BMW X6 SUV 2012',
    'BMW Z4 Convertible 2012',
    'Bentley Arnage Sedan 2009',
    'Bentley Continental Flying Spur Sedan 2007',
    'Bentley Continental GT Coupe 2007',
    'Bentley Continental GT Coupe 2012',
    'Bentley Continental Supersports Conv. Convertible 2012',
    'Bentley Mulsanne Sedan 2011',
    'Bugatti Veyron 16.4 Convertible 2009',
    'Bugatti Veyron 16.4 Coupe 2009',
    'Buick Enclave SUV 2012',
    'Buick Rainier SUV 2007',
    'Buick Regal GS 2012',
    'Buick Verano Sedan 2012',
    'Cadillac CTS-V Sedan 2012',
    'Cadillac Escalade EXT Crew Cab 2007',
    'Cadillac SRX SUV 2012',
    'Chevrolet Avalanche Crew Cab 2012',
    'Chevrolet Camaro Convertible 2012',
    'Chevrolet Cobalt SS 2010',
    'Chevrolet Corvette Convertible 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    'Chevrolet Corvette ZR1 2012',
    'Chevrolet Express Cargo Van 2007',
    'Chevrolet Express Van 2007',
    'Chevrolet HHR SS 2010',
    'Chevrolet Impala Sedan 2007',
    'Chevrolet Malibu Hybrid Sedan 2010',
    'Chevrolet Malibu Sedan 2007',
    'Chevrolet Monte Carlo Coupe 2007',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007',
    'Chevrolet Silverado 1500 Extended Cab 2012',
    'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012',
    'Chevrolet Silverado 2500HD Regular Cab 2012',
    'Chevrolet Sonic Sedan 2012',
    'Chevrolet Tahoe Hybrid SUV 2012',
    'Chevrolet TrailBlazer SS 2009',
    'Chevrolet Traverse SUV 2012',
    'Chrysler 300 SRT-8 2010',
    'Chrysler Aspen SUV 2009',
    'Chrysler Crossfire Convertible 2008',
    'Chrysler PT Cruiser Convertible 2008',
    'Chrysler Sebring Convertible 2010',
    'Chrysler Town and Country Minivan 2012',
    'Daewoo Nubira Wagon 2002',
    'Dodge Caliber Wagon 2007',
    'Dodge Caliber Wagon 2012',
    'Dodge Caravan Minivan 1997',
    'Dodge Challenger SRT8 2011',
    'Dodge Charger SRT-8 2009',
    'Dodge Charger Sedan 2012',
    'Dodge Dakota Club Cab 2007',
    'Dodge Dakota Crew Cab 2010',
    'Dodge Durango SUV 2007',
    'Dodge Durango SUV 2012',
    'Dodge Journey SUV 2012',
    'Dodge Magnum Wagon 2008',
    'Dodge Ram Pickup 3500 Crew Cab 2010',
    'Dodge Ram Pickup 3500 Quad Cab 2009',
    'Dodge Sprinter Cargo Van 2009',
    'Eagle Talon Hatchback 1998',
    'FIAT 500 Abarth 2012',
    'FIAT 500 Convertible 2012',
    'Ferrari 458 Italia Convertible 2012',
    'Ferrari 458 Italia Coupe 2012',
    'Ferrari California Convertible 2012',
    'Ferrari FF Coupe 2012',
    'Fisker Karma Sedan 2012',
    'Ford E-Series Wagon Van 2012',
    'Ford Edge SUV 2012',
    'Ford Expedition EL SUV 2009',
    'Ford F-150 Regular Cab 2007',
    'Ford F-150 Regular Cab 2012',
    'Ford F-450 Super Duty Crew Cab 2012',
    'Ford Fiesta Sedan 2012',
    'Ford Focus Sedan 2007',
    'Ford Freestar Minivan 2007',
    'Ford GT Coupe 2006',
    'Ford Mustang Convertible 2007',
    'Ford Ranger SuperCab 2011',
    'GMC Acadia SUV 2012',
    'GMC Canyon Extended Cab 2012',
    'GMC Savana Van 2012',
    'GMC Terrain SUV 2012',
    'GMC Yukon Hybrid SUV 2012',
    'Geo Metro Convertible 1993',
    'HUMMER H2 SUT Crew Cab 2009',
    'HUMMER H3T Crew Cab 2010',
    'Honda Accord Coupe 2012',
    'Honda Accord Sedan 2012',
    'Honda Odyssey Minivan 2007',
    'Honda Odyssey Minivan 2012',
    'Hyundai Accent Sedan 2012',
    'Hyundai Azera Sedan 2012',
    'Hyundai Elantra Sedan 2007',
    'Hyundai Elantra Touring Hatchback 2012',
    'Hyundai Genesis Sedan 2012',
    'Hyundai Santa Fe SUV 2012',
    'Hyundai Sonata Hybrid Sedan 2012',
    'Hyundai Sonata Sedan 2012',
    'Hyundai Tucson SUV 2012',
    'Hyundai Veloster Hatchback 2012',
    'Hyundai Veracruz SUV 2012',
    'Infiniti G Coupe IPL 2012',
    'Infiniti QX56 SUV 2011',
    'Isuzu Ascender SUV 2008',
    'Jaguar XK XKR 2012',
    'Jeep Compass SUV 2012',
    'Jeep Grand Cherokee SUV 2012',
    'Jeep Liberty SUV 2012',
    'Jeep Patriot SUV 2012',
    'Jeep Wrangler SUV 2012',
    'Lamborghini Aventador Coupe 2012',
    'Lamborghini Diablo Coupe 2001',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    'Lamborghini Reventon Coupe 2008',
    'Land Rover LR2 SUV 2012',
    'Land Rover Range Rover SUV 2012',
    'Lincoln Town Car Sedan 2011',
    'MINI Cooper Roadster Convertible 2012',
    'Maybach Landaulet Convertible 2012',
    'Mazda Tribute SUV 2011',
    'McLaren MP4-12C Coupe 2012',
    'Mercedes-Benz 300-Class Convertible 1993',
    'Mercedes-Benz C-Class Sedan 2012',
    'Mercedes-Benz E-Class Sedan 2012',
    'Mercedes-Benz S-Class Sedan 2012',
    'Mercedes-Benz SL-Class Coupe 2009',
    'Mercedes-Benz Sprinter Van 2012',
    'Mitsubishi Lancer Sedan 2012',
    'Nissan 240SX Coupe 1998',
    'Nissan Juke Hatchback 2012',
    'Nissan Leaf Hatchback 2012',
    'Nissan NV Passenger Van 2012',
    'Plymouth Neon Coupe 1999',
    'Porsche Panamera Sedan 2012',
    'Ram C',
    'Rolls-Royce Ghost Sedan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    'Rolls-Royce Phantom Sedan 2012',
    'Scion xD Hatchback 2012',
    'Spyker C8 Convertible 2009',
    'Spyker C8 Coupe 2009',
    'Suzuki Aerio Sedan 2007',
    'Suzuki Kizashi Sedan 2012',
    'Suzuki SX4 Hatchback 2012',
    'Suzuki SX4 Sedan 2012',
    'Tesla Model S Sedan 2012',
    'Toyota 4Runner SUV 2012',
    'Toyota Camry Sedan 2012',
    'Toyota Corolla Sedan 2012',
    'Toyota Sequoia SUV 2012',
    'Volkswagen Beetle Hatchback 2012',
    'Volkswagen Golf Hatchback 1991',
    'Volkswagen Golf Hatchback 2012',
    'Volvo 240 Sedan 1993',
    'Volvo C30 Hatchback 2012',
    'Volvo XC90 SUV 2007',
    'smart fortwo Convertible 2012'
]

# Validate class names length
if len(CLASS_NAMES) != EXPECTED_CLASSES:
    raise RuntimeError(f"Number of class names ({len(CLASS_NAMES)}) does not match EXPECTED_CLASSES ({EXPECTED_CLASSES})")

# --- Model Definition ---
class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int = EXPECTED_CLASSES):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False)
        in_features: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# --- Helper Functions ---
def load_model(model_path: Path, num_classes: int) -> Tuple[ResNet50Model, torch.device]:
    """Loads the pre-trained model and determines the device."""
    if not model_path.is_file():
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        raise RuntimeError(f"Model file not found: {model_path}")
    try:
        start_time = time.time()
        print(f"Loading model from {model_path}...")
        model = ResNet50Model(num_classes=num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()

        model_output_features = model.backbone.fc.out_features
        if model_output_features != num_classes:
            raise RuntimeError(f"Model loaded has {model_output_features} output features, but expected {num_classes}")

        print(f"Model loaded successfully to {device} in {time.time() - start_time:.2f}s.")
        return model, device
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error loading model: {e}")

def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Loads, preprocesses an image, and moves it to the specified device."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found at '{image_path}'")
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor.to(device)
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}", file=sys.stderr)
        raise RuntimeError(f"Error processing image: {e}")

def predict_single_image(
    image_path: str,
    model: nn.Module,
    class_names: List[str],
    device: torch.device
) -> Tuple[str, Dict[str, float]]:
    """Performs prediction on a single image using the loaded model."""
    try:
        start_time = time.time()
        print(f"Predicting for: {image_path}")
        image_tensor = preprocess_image(image_path, device)
        with torch.no_grad():
            outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]
        probability_dict = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
        print(f"Prediction finished in {time.time() - start_time:.2f}s. Result: {predicted_class}")
        return predicted_class, probability_dict
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        print(f"Error during prediction for '{image_path}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error during prediction: {e}")

# --- FastAPI App Setup ---
app = FastAPI(title="Image Classification API", version="1.0.0")

# --- Application State ---
app_state: Dict = {
    "model": None,
    "class_names": None,
    "device": None,
    "is_ready": False,
    "startup_error": None
}

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Tải model và class names khi server khởi động."""
    print("Server starting up. Loading resources...")
    try:
        app_state["class_names"] = CLASS_NAMES
        app_state["model"], app_state["device"] = load_model(MODEL_PATH, EXPECTED_CLASSES)
        app_state["is_ready"] = True
        app_state["startup_error"] = None
        print(f"--- Server is ready to accept requests on device: {app_state['device']} ---")
    except Exception as e:
        app_state["startup_error"] = str(e)
        app_state["is_ready"] = False
        print(f"--- FATAL STARTUP ERROR: {e} ---", file=sys.stderr)
        print("--- Server failed to load model or class names. Predictions will fail. ---", file=sys.stderr)

# --- Pydantic Models for Request/Response ---
class ImagePathRequest(BaseModel):
    image_path: str

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    message: Optional[str] = None
    model_loaded: bool
    device: Optional[str] = None

# --- API Endpoints ---
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái của server và model."""
    device_name = str(app_state['device']) if app_state['device'] else None
    if app_state["is_ready"]:
        return HealthCheckResponse(status="OK", model_loaded=True, device=device_name)
    else:
        return HealthCheckResponse(
            status="ERROR",
            message=f"Server startup failed: {app_state['startup_error']}" if app_state['startup_error'] else "Model not ready.",
            model_loaded=False,
            device=device_name
        )

@app.post("/predict_image/", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(request: ImagePathRequest):
    """Nhận đường dẫn ảnh (tuyệt đối) và trả về kết quả dự đoán."""
    if not app_state["is_ready"] or app_state["model"] is None or app_state["class_names"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model or class names not loaded. Startup error: {app_state['startup_error']}" if app_state['startup_error'] else "Service not ready."
        )
    try:
        prediction, probabilities = predict_single_image(
            image_path=request.image_path,
            model=app_state["model"],
            class_names=app_state["class_names"],
            device=app_state["device"]
        )
        return PredictionResponse(prediction=prediction, probabilities=probabilities)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at path: {request.image_path}"
        )
    except Exception as e:
        print(f"Internal server error during prediction: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}"
        )

# --- Run the server ---
if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    print(f"Model path: {MODEL_PATH}")
    print("To run the API server, use the command from the project root directory:")
    print(f"'{sys.executable}' -m uvicorn model_api:app --host 127.0.0.1 --8000")
    print("(Add --reload for development, remove for production)")