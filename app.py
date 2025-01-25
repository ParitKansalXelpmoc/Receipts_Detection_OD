from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io
import torch
import torchvision.transforms.functional as F
import numpy as np

# Custom imports
from src.pipeline import BillRoiPredictor
from config import MODEL_PATH, CONFIDENCE_THRESHOLD
from src.post_processing import iou, merge_boxes_iteratively

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = BillRoiPredictor(model_path=MODEL_PATH)  # Update with your actual model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionResponse(BaseModel):
    boxes: list
    scores: list
    labels: list


# Load and preprocess image
def load_image(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return F.to_tensor(pil_image).to(device)


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    """
    Predict bounding boxes for the uploaded image.
    """
    try:
        # Read image bytes and convert to PIL Image
        image_bytes = await image.read()
        
        # Load image as tensor
        image_tensor = load_image(image_bytes)

        # Perform prediction
        with torch.no_grad():
            prediction = predictor.predict_image(image_tensor)

        boxes = prediction[0]["boxes"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()

        # Apply confidence threshold to filter valid detections
        valid_detections = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        labels = labels[valid_detections]

        # Merge boxes using IoU
        merged_boxes = merge_boxes_iteratively(boxes, iou_threshold=0.1)

        # Return the results
        return {
            "boxes": merged_boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
