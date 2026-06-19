import os
import cv2
import json
import time
import torch
import numpy as np
import torchvision
from torchvision.transforms import functional as F

# =========================================================
# Paths
# =========================================================

base_dir = "/home/amir-fx507/MaskRCNNtest"

model_path = os.path.join(base_dir, "mask_rcnn_best.pth")
image_path = os.path.join(base_dir, "frame656.jpg")
annotations_path = os.path.join(base_dir, "annotations.json")

# =========================================================
# Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

# =========================================================
# Load Classes
# =========================================================

with open(annotations_path, "r") as f:
    ann = json.load(f)

categories = ann["categories"]

class_names = [c["name"] for c in categories]

num_classes = len(class_names) + 1  # + background

print("\nClasses:")
print(class_names)

# =========================================================
# Create Model
# =========================================================

model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

# Replace box predictor
in_features_box = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = \
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box,
        num_classes
    )

# Replace mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256

model.roi_heads.mask_predictor = \
    torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

# =========================================================
# Load Weights
# =========================================================

checkpoint = torch.load(model_path, map_location=device)

# Support both direct state_dict and checkpoint dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)

model.eval()

print("\nModel loaded successfully")

# =========================================================
# Load Image
# =========================================================

img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError(image_path)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# Preprocess
# =========================================================

img_tensor = F.to_tensor(img_rgb).to(device)

# =========================================================
# Inference
# =========================================================

with torch.no_grad():

    torch.cuda.synchronize()

    start = time.time()

    outputs = model([img_tensor])

    torch.cuda.synchronize()

    end = time.time()

print(f"\nInference time: {(end-start)*1000:.2f} ms")

# =========================================================
# Parse Outputs
# =========================================================

output = outputs[0]

boxes = output["boxes"].detach().cpu().numpy()
labels = output["labels"].detach().cpu().numpy()
scores = output["scores"].detach().cpu().numpy()
masks = output["masks"].detach().cpu().numpy()

print(f"Detections: {len(boxes)}")

# =========================================================
# Visualization
# =========================================================

conf_threshold = 0.3

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

img_out = img_rgb.copy()

for i in range(len(scores)):

    score = scores[i]

    if score < conf_threshold:
        continue

    label = labels[i]

    if label <= 0:
        continue

    x1, y1, x2, y2 = boxes[i].astype(int)

    # =====================================================
    # Mask
    # =====================================================

    mask = masks[i, 0]

    mask = mask > 0.5

    # =====================================================
    # Color
    # =====================================================

    color = colors[i % len(colors)]

    overlay = np.array(color, dtype=np.uint8)

    img_out[mask] = (
        0.5 * img_out[mask] +
        0.5 * overlay
    ).astype(np.uint8)

    # =====================================================
    # Label Name
    # =====================================================

    label_index = label - 1

    if 0 <= label_index < len(class_names):
        label_name = class_names[label_index]
    else:
        label_name = "Unknown"

    # =====================================================
    # Draw Box
    # =====================================================

    cv2.rectangle(
        img_out,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        img_out,
        f"{label_name} {score:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# =========================================================
# Save Result
# =========================================================

output_path = os.path.join(base_dir, "result_pytorch.jpg")

cv2.imwrite(
    output_path,
    cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
)

print(f"\nSaved result to:\n{output_path}")