# track.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor
import resources
import os

# ---------------------------
# Config
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19
OUTPUT_STRIDE = 16
# Use relative path that works in Docker and locally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
ROAD_CLASS_ID = 0
INFER_W, INFER_H = 1024, 512

# ---------------------------
# Load model
# ---------------------------
def load_segmentation_model():
    model = resources.deeplabv3plus_resnet101(
        num_classes=NUM_CLASSES,
        output_stride=OUTPUT_STRIDE,
        pretrained_backbone=False
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ---------------------------
# Preprocessing
# ---------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ---------------------------
# Segmentation functions
# ---------------------------
def get_road_mask(frame_bgr, model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    tensor = to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    pred_fullres = cv2.resize(pred.astype(np.uint8),
                              (frame_bgr.shape[1], frame_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    road_mask = (pred_fullres == ROAD_CLASS_ID).astype(np.uint8) * 255
    return road_mask

def apply_mask(frame_bgr, road_mask):
    mask_colored = np.zeros_like(frame_bgr)
    mask_colored[:, :, 1] = road_mask
    blended = cv2.addWeighted(frame_bgr, 0.7, mask_colored, 0.3, 0)
    return blended

# ---------------------------
# Polygon utilities
# ---------------------------
def fit_line(points):
    if len(points) < 2:
        return None
    X = points[:,0].reshape(-1,1)
    y = points[:,1]
    model = RANSACRegressor().fit(X, y)
    a = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    return a, b

def get_line_endpoints(a, b, contour, side="left"):
    xs = contour[:,0,0]
    if side=="left":
        edge_pts = contour[xs < np.median(xs)][:,0]
    else:
        edge_pts = contour[xs >= np.median(xs)][:,0]

    if len(edge_pts) == 0:
        return None, None
    top = tuple(edge_pts[edge_pts[:,1].argmin()])
    bottom = tuple(edge_pts[edge_pts[:,1].argmax()])
    return top, bottom

def get_road_polygon(road_mask):
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)

    xs = c[:,0,0]
    midpoint = np.median(xs)
    left_pts = c[xs < midpoint][:,0]
    right_pts = c[xs >= midpoint][:,0]

    aL, bL = fit_line(left_pts)
    aR, bR = fit_line(right_pts)

    top_left, bottom_left = get_line_endpoints(aL, bL, c, "left")
    top_right, bottom_right = get_line_endpoints(aR, bR, c, "right")

    if None in [top_left, bottom_left, top_right, bottom_right]:
        return None

    polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    return polygon

def draw_polygon(frame_bgr, polygon_pts):
    if polygon_pts is None: 
        return frame_bgr
    
    # Ensure polygon is int32 for OpenCV
    if not isinstance(polygon_pts, np.ndarray):
        polygon_pts = np.array(polygon_pts, dtype=np.int32)
    elif polygon_pts.dtype != np.int32:
        polygon_pts = polygon_pts.astype(np.int32)
    
    overlay = frame_bgr.copy()
    cv2.polylines(overlay, [polygon_pts], isClosed=True, color=(0,0,255), thickness=3)
    return overlay
