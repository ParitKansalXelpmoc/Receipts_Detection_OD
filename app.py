from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import io
import torch
import base64
import numpy as np

# Import the predictor class from src.pipeline
from src.pipeline import BillRoiPredictor
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = BillRoiPredictor(model_path=MODEL_PATH)  # Update with your actual model path

# Function to compute Intersection over Union (IOU)
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # If no intersection, return 0
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# Function to combine two boxes
def combine_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    combined_x_min = min(x1_min, x2_min)
    combined_y_min = min(y1_min, y2_min)
    combined_x_max = max(x1_max, x2_max)
    combined_y_max = max(y1_max, y2_max)
    
    return [combined_x_min, combined_y_min, combined_x_max, combined_y_max]

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
        
        # Combine bounding boxes based on IOU_THRESHOLD
        combined_boxes = []
        combined_scores = []
        combined_labels = []
        
        for i in range(len(boxes)):
            box1 = boxes[i]
            score1 = scores[i]
            label1 = labels[i]
            
            is_combined = False
            for j in range(len(combined_boxes)):
                box2 = combined_boxes[j]
                
                # If IOU between the boxes is above the threshold, combine them
                if compute_iou(box1, box2) > IOU_THRESHOLD:
                    combined_box = combine_boxes(box1, box2)
                    combined_boxes[j] = combined_box
                    combined_scores[j] = max(score1, combined_scores[j])  # Take the highest score
                    combined_labels[j] = label1  # Could also keep the majority label
                    is_combined = True
                    break
            
            if not is_combined:
                combined_boxes.append(box1)
                combined_scores.append(score1)
                combined_labels.append(label1)
        
        # Extract cropped images and convert to byte form
        cropped_images = []
        for box in combined_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))
            img_bytes_io = io.BytesIO()
            cropped_img.save(img_bytes_io, format="PNG")
            cropped_image_bytes = base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")
            cropped_images.append(cropped_image_bytes)
        
        return {
            "boxes": combined_boxes,
            "scores": combined_scores,
            "labels": combined_labels,
            "cropped_images": cropped_images  # Base64 encoded cropped images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
