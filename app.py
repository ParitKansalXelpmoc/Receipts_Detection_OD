from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import io
import torch

# Import the predictor class
from src.pipeline import BillRoiPredictor
from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = BillRoiPredictor(model_path=MODEL_PATH)  # Update with your actual model path

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
        predictions = predictor.predict_image(pil_image)

        # Extract bounding boxes, scores, labels, and masks
        boxes = predictions[0]["boxes"] if "boxes" in predictions[0] else torch.tensor([])
        scores = predictions[0]["scores"] if "scores" in predictions[0] else torch.tensor([])
        labels = predictions[0]["labels"] if "labels" in predictions[0] else torch.tensor([])
        masks = predictions[0]["masks"] if "masks" in predictions[0] else torch.tensor([])

        # Apply confidence threshold to filter valid detections
        valid_detections = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections].tolist()
        scores = scores[valid_detections].tolist()
        labels = labels[valid_detections].tolist()
        masks = masks[valid_detections].tolist()

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
