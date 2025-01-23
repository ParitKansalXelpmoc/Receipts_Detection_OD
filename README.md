# Receipts_Detection_OD

### CROP IMAGE
```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://127.0.0.1:5000/predict"
image_path = r"image_path"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files).json()

boxes = response["boxes"]
cropped_images = response["cropped_images"]

# Decode and save/display each cropped image
for i, (box, cropped_image_b64) in enumerate(zip(boxes, cropped_images)):
    cropped_image_data = base64.b64decode(cropped_image_b64)
    cropped_image = Image.open(BytesIO(cropped_image_data))
    cropped_image.show()  # Display cropped image
    cropped_image.save(f"cropped_{i}.png")  # Save cropped image
```
