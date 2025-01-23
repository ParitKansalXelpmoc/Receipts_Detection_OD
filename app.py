from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import io
import torch
import base64

# Import the predictor class
from src.pipeline import BillRoiPredictor
from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = BillRoiPredictor(model_path=MODEL_PATH)  # Update with your actual model path

class CroppedImageResponse(BaseModel):
    boxes: list
    scores: list
    labels: list
    cropped_images: list  # List of cropped images in byte form

@app.post("/predict", response_model=CroppedImageResponse)
async def predict(image: UploadFile = File(...)):
    """
    Predict bounding boxes for the uploaded image and return cropped images.
    """
    try:
        # Read image bytes and convert to PIL Image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Perform prediction
        predictions = predictor.predict_image(pil_image)

        # Extract bounding boxes, scores, labels
        boxes = predictions[0].get("boxes", torch.tensor([]))
        scores = predictions[0].get("scores", torch.tensor([]))
        labels = predictions[0].get("labels", torch.tensor([]))

        # Apply confidence threshold to filter valid detections
        valid_detections = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections].tolist()
        scores = scores[valid_detections].tolist()
        labels = labels[valid_detections].tolist()
        
        # Extract cropped images and convert to byte form
        cropped_images = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))
            img_bytes_io = io.BytesIO()
            cropped_img.save(img_bytes_io, format="PNG")
            cropped_image_bytes = base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")
            cropped_images.append(cropped_image_bytes)

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "cropped_images": cropped_images  # Base64 encoded cropped images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
