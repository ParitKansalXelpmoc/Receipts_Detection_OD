# Receipts_Detection_OD

### CROP IMAGE
```python

url = "http://127.0.0.1:5000/predict"
image_path = r"image_path"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

boxes = response["boxes"]

# Crop and save/display each bounding box
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    cropped_image = image.crop((x1, y1, x2, y2))  # Crop the bounding box


```