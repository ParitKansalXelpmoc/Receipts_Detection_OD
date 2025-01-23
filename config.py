import torch, os

VERSION = "receipt_v2.0"
ROOT = os.getcwd()
# ROOT = r"D:\project\FSL\new_codebase\FSL_UB_API"
artifact = 'artifacts'
LOG_DIR = "logs"
LOG_FILE = "model_logs.log"
LOGFILE_DIR = os.path.join(ROOT, LOG_DIR, LOG_FILE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = os.path.join(ROOT, artifact, "model_9.pth")
CONFIDENCE_THRESHOLD = 0.99
IOU_THRESHOLD = 0.1
