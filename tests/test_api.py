import requests

url = "http://127.0.0.1:5000/predict"
image_path = r"D:\project\Receipts_Detection_OD\artifacts\val_test_2.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
