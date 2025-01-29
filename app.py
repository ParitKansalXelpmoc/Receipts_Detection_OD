from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

# Custom imports
from src.pipeline import BillRoiPredictor
from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = BillRoiPredictor(model_path=MODEL_PATH)  # Update with your actual model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionResponse(BaseModel):
    boxes: list
    scores: list
    labels: list


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    """
    Predict bounding boxes for the uploaded image.
    """
    try:
        # Read image bytes and convert to PIL Image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Perform prediction
        with torch.no_grad():
            boxes, scores, labels = predictor.predict_image(pil_image)

        # Return the results
        return {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
