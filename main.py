from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import uvicorn

app = FastAPI(title="Plant Disease Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionaries to store models and class names
models = {}
class_names = {}

# Load MobileNetV2 model for feature extraction
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to preprocess images
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = mobilenet_preprocess(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Load models and label encoders on startup
@app.on_event("startup")
async def load_models():
    global models, class_names
    plant_types = ["Potato", "Tomato", "Apple"]  # Add more as needed

    for plant_type in plant_types:
        models[plant_type] = {}
        class_names[plant_type] = []

        plant_dir = os.path.join(os.path.dirname(__file__), 'model', plant_type)

        # Load SVM model
        svm_model_path = os.path.join(plant_dir, f"svm_model_mobilenetv2{('_' + plant_type.lower())}.joblib")
        if os.path.exists(svm_model_path):
            models[plant_type]["svm"] = joblib.load(svm_model_path)
        else:
            print(f"SVM model not found for {plant_type} at {svm_model_path}")
            continue

        # Load label encoder
        label_encoder_path = os.path.join(plant_dir, f"label_encoder_mobilenetv2{('_' + plant_type.lower())}.joblib")
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            models[plant_type]["label_encoder"] = label_encoder
            class_names[plant_type] = label_encoder.classes_.tolist()
        else:
            print(f"Label encoder not found for {plant_type} at {label_encoder_path}")
            continue

@app.get("/health")
async def health_check():
    available_models = {
        plant_type: list(models_dict.keys())
        for plant_type, models_dict in models.items()
        if models_dict
    }
    return {
        "status": "ok",
        "available_models": available_models
    }

@app.get("/models")
async def get_models():
    available_models = {}
    for plant_type, models_dict in models.items():
        if models_dict:
            available_models[plant_type] = {
                "models": list(models_dict.keys()),
                "class_names": class_names.get(plant_type, [])
            }
    return available_models

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    plant_type: str = Form(...)
):
    if plant_type not in models:
        raise HTTPException(status_code=400, detail=f"Unsupported plant type '{plant_type}'. Available types: {list(models.keys())}")

    if "svm" not in models[plant_type] or "label_encoder" not in models[plant_type]:
        raise HTTPException(status_code=500, detail=f"Models for '{plant_type}' are not properly loaded.")

    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        # Extract features using MobileNetV2
        features = mobilenet_model.predict(input_tensor)

        # Predict using SVM model
        svm_model = models[plant_type]["svm"]
        label_encoder = models[plant_type]["label_encoder"]

        predicted_class_index = svm_model.predict(features)[0]
        class_name = label_encoder.inverse_transform([predicted_class_index])[0]

        # Get prediction probabilities if available
        if hasattr(svm_model, "predict_proba"):
            probs = svm_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[::-1][:5]
            results = [
                {
                    "class_index": int(i),
                    "probability": float(probs[i]),
                    "class_name": label_encoder.inverse_transform([i])[0]
                } for i in top_indices
            ]
        else:
            results = [{
                "class_index": int(predicted_class_index),
                "probability": None,
                "class_name": class_name
            }]

        return {
            "status": "success",
            "plant_type": plant_type,
            "model_used": "svm",
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
