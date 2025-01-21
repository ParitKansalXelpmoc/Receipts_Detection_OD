import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# API Endpoint and Input Image Path
url = "http://127.0.0.1:5000/predict"
image_path = r"D:\project\Receipts_Detection_OD\artifacts\val_test.png"

# Send the POST Request
with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

# Check Response Status
if response.status_code == 200:
    response_data = response.json()
    boxes = response_data["boxes"]
    scores = response_data["scores"]

    # Open the Image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Plot Each Box with Confidence Score
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{score:.2f}", fill="red")

    # Display the Image with Bounding Boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
else:
    print(f"Failed to get prediction: {response.status_code}, {response.text}")
