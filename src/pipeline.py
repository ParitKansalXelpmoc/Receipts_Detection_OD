import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from config import *
from src.post_processing import iou, merge_boxes_iteratively

class BillRoiPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.num_classes = 2
        self.model = self._load_model()

    def _load_model(self):
        try:
            # Load a pre-trained Mask R-CNN model
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

            # Replace the box predictor (for bounding boxes)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            # Replace the mask predictor (for segmentation masks)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)

            # Load custom weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            # Move model to the appropriate device
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _get_transforms(self, image):
        return F.to_tensor(image)

    def predict_image(self, image):
        # Convert PIL image to tensor and add batch dimension
        image_tensor = self._get_transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        # Apply confidence threshold to filter valid detections
        valid_detections = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        labels = labels[valid_detections]

        # Merge boxes using IoU
        boxes = merge_boxes_iteratively(boxes, iou_threshold=0.1)
        return boxes, scores, labels
