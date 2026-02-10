"""
Perspective-aware calibration using vehicle dimensions.
This module uses detected vehicles to calibrate real-world distances accounting for camera perspective.

Supports two methods:
1. Vehicle-based exponential model (fallback, no manual calibration needed)
2. Homography-based IPM (Inverse Perspective Mapping) - accurate, requires 4-point calibration
"""

import cv2
import numpy as np
import base64
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Common vehicle dimensions (length in meters) - can be expanded
VEHICLE_DIMENSIONS = {
    "sedan": {"length": 4.5, "width": 1.8, "height": 1.5},
    "suv": {"length": 4.8, "width": 2.0, "height": 1.7},
    "hatchback": {"length": 4.0, "width": 1.7, "height": 1.5},
    "truck": {"length": 5.5, "width": 2.1, "height": 2.0},
    "bus": {"length": 12.0, "width": 2.5, "height": 3.0},
    "motorcycle": {"length": 2.2, "width": 0.8, "height": 1.2},
    "van": {"length": 5.0, "width": 2.0, "height": 2.2},
}


def calculate_distance_with_homography(line_a_y, line_b_y, img_width, img_height, homography_matrix=None):
    """
    Calculate real-world distance between two lines using homography transformation (IPM).
    This is the most accurate method for elevated/angled cameras.
    
    Based on standard traffic surveillance literature (e.g., ITSC papers on IPM).
    
    Args:
        line_a_y: Y-coordinate of line A (top line)
        line_b_y: Y-coordinate of line B (bottom line)  
        img_width: Image width in pixels
        img_height: Image height in pixels
        homography_matrix: 3x3 numpy array for image→road plane transformation.
                          If None, must be computed first via 4-point calibration.
    
    Returns:
        dict: {
            'distance_meters': float,
            'method': 'homography_ipm',
            'confidence': float,
            'warning': str (if any)
        }
    """
    if homography_matrix is None:
        return {
            'distance_meters': None,
            'method': 'homography_ipm',
            'confidence': 0.0,
            'warning': 'Homography matrix not calibrated. Use 4-point calibration first.'
        }
    
    try:
        # Take center points of each line in image coordinates
        # Line A (top, entry)
        point_a_img = np.array([[img_width / 2, line_a_y]], dtype=np.float32)
        # Line B (bottom, exit)
        point_b_img = np.array([[img_width / 2, line_b_y]], dtype=np.float32)
        
        # Transform both points to road plane using homography
        # cv2.perspectiveTransform requires shape (N, 1, 2)
        point_a_img = point_a_img.reshape(-1, 1, 2)
        point_b_img = point_b_img.reshape(-1, 1, 2)
        
        point_a_world = cv2.perspectiveTransform(point_a_img, homography_matrix)
        point_b_world = cv2.perspectiveTransform(point_b_img, homography_matrix)
        
        # Extract coordinates on road plane (in meters)
        xa, ya = point_a_world[0, 0]
        xb, yb = point_b_world[0, 0]
        
        # Calculate Euclidean distance on road plane
        distance_meters = np.sqrt((xb - xa)**2 + (yb - ya)**2)
        
        print(f"[HomographyIPM] Line A: image({img_width/2:.0f}, {line_a_y:.0f}) → road({xa:.2f}, {ya:.2f})m")
        print(f"[HomographyIPM] Line B: image({img_width/2:.0f}, {line_b_y:.0f}) → road({xb:.2f}, {yb:.2f})m")
        print(f"[HomographyIPM] ✅ Distance: {distance_meters:.2f}m (Euclidean on road plane)")
        
        return {
            'distance_meters': distance_meters,
            'method': 'homography_ipm',
            'confidence': 0.95,  # High confidence with proper homography
            'points_world': {'A': (xa, ya), 'B': (xb, yb)}
        }
        
    except Exception as e:
        print(f"[HomographyIPM] ❌ Error: {e}")
        return {
            'distance_meters': None,
            'method': 'homography_ipm',
            'confidence': 0.0,
            'warning': f'Homography transformation failed: {str(e)}'
        }


def estimate_homography_from_points(image_points, world_points):
    """
    Estimate homography matrix from 4 corresponding points.
    
    Args:
        image_points: 4x2 numpy array of (x,y) in image coordinates (pixels)
        world_points: 4x2 numpy array of (x,y) in real-world coordinates (meters)
    
    Returns:
        3x3 homography matrix (numpy array), or None if failed
        
    Example:
        # User clicks 4 corners of a known rectangle on the road (e.g., parking space, lane markings)
        # Image coordinates: [(100, 500), (900, 500), (950, 300), (50, 300)]  
        # Real-world: [(0, 0), (5, 0), (5, 10), (0, 10)]  # 5m wide, 10m long rectangle
        H = estimate_homography_from_points(img_pts, world_pts)
    """
    if len(image_points) != 4 or len(world_points) != 4:
        print("[Homography] ❌ Need exactly 4 point pairs")
        return None
    
    try:
        # Convert to proper format
        src_pts = np.array(image_points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        
        # Compute homography using OpenCV
        H, status = cv2.findHomography(src_pts, dst_pts, method=0)  # 0 = all points used
        
        if H is None:
            print("[Homography] ❌ Failed to compute matrix")
            return None
            
        print("[Homography] ✅ Matrix computed:")
        print(H)
        return H
        
    except Exception as e:
        print(f"[Homography] ❌ Error: {e}")
        return None


def get_vehicle_analysis(image_path, bbox, gemini_api_key):
    """
    Analyze a vehicle using Gemini to determine:
    1. Vehicle type/model
    2. Orientation (angle relative to camera)
    3. Visible dimensions in the image
    
    Args:
        image_path: Path to the frame image
        bbox: Bounding box [x1, y1, x2, y2] of the vehicle
        gemini_api_key: API key for Gemini
    
    Returns:
        dict with vehicle_type, length_pixels, orientation_angle, confidence
    """
    print(f"[PerspectiveCalib] Analyzing vehicle at bbox: {bbox}")
    
    # Load image and crop to vehicle
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_height, img_width = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Ensure bbox is within image bounds
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_width, int(x2)), min(img_height, int(y2))
    
    # Draw bbox and measurement lines on full image for context
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add center line for length measurement
    center_y = (y1 + y2) // 2
    cv2.line(annotated_img, (x1, center_y), (x2, center_y), (255, 0, 0), 2)
    
    # Add pixel measurement text
    vehicle_width_pixels = x2 - x1
    vehicle_height_pixels = y2 - y1
    cv2.putText(annotated_img, f"Width: {vehicle_width_pixels}px", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Encode image
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Prepare Gemini prompt - SIMPLIFIED to reduce token usage
    system_prompt = (
        f"Analyze vehicle in green box. Image: {img_width}x{img_height}px. Vehicle box: {vehicle_width_pixels}x{vehicle_height_pixels}px.\n"
        f"Return JSON only (no markdown):\n"
        f'{{"vehicle_type": "sedan/suv/hatchback/truck/bus/motorcycle/van", "orientation_angle": 0-90, "primary_dimension": "length/width", "confidence": 0.0-1.0, "reasoning": "1 sentence"}}'
    )
    
    try:
        chat = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_output_tokens=2048,  # Increased to accommodate reasoning tokens
        )
        
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this vehicle and return ONLY the JSON object (no markdown, no explanations)."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        )
        
        print(f"[PerspectiveCalib] Sending vehicle analysis request...")
        response = chat.invoke([SystemMessage(content=system_prompt), human_message])
        text = response.content if hasattr(response, 'content') else str(response)
        print(f"[PerspectiveCalib] Gemini vehicle analysis response: '{text}'")
        
        # Check if response is empty
        if not text or not text.strip():
            print(f"[PerspectiveCalib] ERROR: Empty response from Gemini")
            # Return default fallback
            return {
                "vehicle_type": "sedan",
                "orientation_angle": 0,
                "primary_dimension": "length",
                "confidence": 0.5,
                "reasoning": "Default fallback due to empty Gemini response",
                "bbox_width_pixels": vehicle_width_pixels,
                "bbox_height_pixels": vehicle_height_pixels,
                "bbox": bbox
            }
        
        # Parse JSON
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = "\n".join(cleaned_text.splitlines()[1:])
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        if '{' in cleaned_text and '}' in cleaned_text:
            json_str = cleaned_text[cleaned_text.find('{'):cleaned_text.rfind('}')+1]
            result = json.loads(json_str)
            
            # Add pixel measurements
            result['bbox_width_pixels'] = vehicle_width_pixels
            result['bbox_height_pixels'] = vehicle_height_pixels
            result['bbox'] = bbox
            
            return result
    except Exception as e:
        print(f"[PerspectiveCalib] Error analyzing vehicle: {e}")
        return None


def calculate_perspective_calibration(image_path, line_a_y, line_b_y, vehicle_bbox, gemini_api_key, homography_matrix=None):
    """
    Calculate real-world distance between calibration lines.
    
    Uses two methods (in priority order):
    1. Homography/IPM: If homography_matrix provided, uses inverse perspective mapping (most accurate)
    2. Vehicle-based: Falls back to vehicle dimension analysis with exponential perspective model
    
    Args:
        image_path: Path to frame with vehicle
        line_a_y: Y-coordinate of line A (top)
        line_b_y: Y-coordinate of line B (bottom)
        vehicle_bbox: [x1, y1, x2, y2] of vehicle (needed for fallback method)
        gemini_api_key: API key for Gemini (needed for fallback method)
        homography_matrix: Optional 3x3 numpy array for IPM. If provided, method 1 is used.
    
    Returns:
        dict with distance_meters, confidence, method, details
    """
    print(f"[PerspectiveCalib] Starting calibration with lines A={line_a_y}, B={line_b_y}")
    
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        return {"distance_meters": None, "confidence": 0.0, "note": "Failed to load image"}
    
    img_height, img_width = img.shape[:2]
    
    # METHOD 1: Homography-based IPM (PREFERRED - from ITSC paper approach)
    if homography_matrix is not None:
        print(f"[PerspectiveCalib] ✅ Using HOMOGRAPHY/IPM method (most accurate)")
        result = calculate_distance_with_homography(
            line_a_y, line_b_y, img_width, img_height, homography_matrix
        )
        if result.get('distance_meters') is not None:
            return result
        else:
            print(f"[PerspectiveCalib] ⚠️ Homography failed, falling back to vehicle-based method")
    
    # METHOD 2: Vehicle-based calibration (FALLBACK)
    print(f"[PerspectiveCalib] Using VEHICLE-BASED exponential perspective method")
    
    # Analyze vehicle
    vehicle_info = get_vehicle_analysis(image_path, vehicle_bbox, gemini_api_key)
    if not vehicle_info:
        return {"distance_meters": None, "confidence": 0.0, "note": "Vehicle analysis failed"}
    
    print(f"[PerspectiveCalib] Vehicle identified: {vehicle_info}")
    
    # Get known vehicle dimensions
    vehicle_type = vehicle_info.get('vehicle_type', 'sedan').lower()
    if vehicle_type not in VEHICLE_DIMENSIONS:
        # Try to match partial name
        for vtype in VEHICLE_DIMENSIONS:
            if vtype in vehicle_type or vehicle_type in vtype:
                vehicle_type = vtype
                break
        else:
            vehicle_type = 'sedan'  # Default fallback
    
    real_dimensions = VEHICLE_DIMENSIONS[vehicle_type]
    print(f"[PerspectiveCalib] Using dimensions for {vehicle_type}: {real_dimensions}")
    
    # Determine which dimension to use based on orientation
    primary_dim = vehicle_info.get('primary_dimension', 'length')
    orientation_angle = vehicle_info.get('orientation_angle', 0)
    
    # IMPORTANT FIX: For perspective calibration, we need the dimension ALONG the road direction
    # If primary_dimension is 'width', it means vehicle is viewed from front/rear
    # In this case, the bbox HEIGHT represents the vehicle WIDTH in image
    # The bbox WIDTH represents how much road space the vehicle occupies
    
    if primary_dim == 'length':
        # Side view: vehicle length is visible horizontally
        real_dimension_m = real_dimensions['length']
        pixel_dimension = vehicle_info['bbox_width_pixels']
        # Adjust for angle: if not perpendicular, apparent length is foreshortened
        angle_factor = abs(np.cos(np.radians(orientation_angle)))
        real_dimension_m *= max(angle_factor, 0.5)  # Don't reduce too much
        dimension_label = "length"
    else:  # width
        # Front/rear view: vehicle width is visible
        # For calibration, we want the dimension ALONG the road (perpendicular to camera)
        # Use VEHICLE LENGTH but measured as bbox HEIGHT (depth in perspective)
        # This is because we're measuring distance along the road, not across it
        real_dimension_m = real_dimensions['length']  # Use length for road distance
        # The bbox height in rear/front view represents depth (vehicle length in perspective)
        pixel_dimension = vehicle_info['bbox_height_pixels']
        angle_factor = 1.0
        dimension_label = "length (from rear/front view)"
        
        print(f"[PerspectiveCalib] 🔧 CORRECTION: Vehicle in front/rear view, using LENGTH for road distance measurement")
    
    print(f"[PerspectiveCalib] Using {dimension_label}: {real_dimension_m:.1f}m over {pixel_dimension}px (angle factor: {angle_factor:.2f})")
    
    # Calculate pixel-to-meter ratio at vehicle position
    vehicle_center_y = (vehicle_bbox[1] + vehicle_bbox[3]) / 2
    pixels_per_meter_at_vehicle = pixel_dimension / real_dimension_m
    
    print(f"[PerspectiveCalib] At vehicle position (y={vehicle_center_y:.1f}): {pixels_per_meter_at_vehicle:.2f} pixels/meter")
    
    # Account for perspective: objects farther from camera (lower y) appear smaller
    # For elevated, angled cameras, we need a more sophisticated model
    img = cv2.imread(image_path)
    img_height = img.shape[0]
    
    # Calculate distance in pixels between lines
    line_distance_pixels = abs(line_b_y - line_a_y)
    
    # ENHANCED PERSPECTIVE MODEL for elevated cameras
    # Key insight: pixel-to-meter ratio changes NON-LINEARLY with y-position
    # Objects at bottom of frame are close and large, top are far and small
    
    # Estimate perspective parameters from vehicle measurement
    # Assume exponential perspective model: scale = scale_at_vehicle * exp(k * (y - y_vehicle))
    # where k is the perspective decay constant
    
    # For elevated highway cameras, typical k values: 0.001 to 0.003 per pixel
    # Higher values = steeper perspective (more elevated camera)
    # AUTO-DETECT perspective strength from vehicle position and size
    
    # If vehicle is in lower half of image (closer to camera), use steeper decay
    y_ratio = vehicle_center_y / img_height
    if y_ratio > 0.7:  # Bottom 30% - very close to camera
        perspective_decay = 0.0025  # Steep perspective
        print(f"[PerspectiveCalib] 📐 Vehicle near bottom (y={y_ratio:.1%}) - using STEEP perspective (k=0.0025)")
    elif y_ratio > 0.5:  # Middle region
        perspective_decay = 0.002  # Moderate perspective  
        print(f"[PerspectiveCalib] 📐 Vehicle in middle (y={y_ratio:.1%}) - using MODERATE perspective (k=0.002)")
    else:  # Upper half - farther from camera
        perspective_decay = 0.0015  # Gentler perspective
        print(f"[PerspectiveCalib] 📐 Vehicle far away (y={y_ratio:.1%}) - using GENTLE perspective (k=0.0015)")
    
    # Additional factor: if lines span large y-range, increase decay
    line_span_ratio = line_distance_pixels / img_height
    if line_span_ratio > 0.4:  # Lines span > 40% of image
        perspective_decay *= 1.3
        print(f"[PerspectiveCalib] 📐 Wide line span ({line_span_ratio:.1%}) - increasing decay by 30%")
    
    # Calculate the integral of pixel-to-meter ratio between the two lines
    # This accounts for the changing scale across the distance
    def pixels_per_meter_at_y(y):
        """Calculate pixels per meter at given y coordinate using exponential model"""
        y_diff = y - vehicle_center_y
        scale_factor = np.exp(perspective_decay * y_diff)
        return pixels_per_meter_at_vehicle * scale_factor
    
    # Method selection based on vehicle position
    if abs(vehicle_center_y - line_a_y) < line_distance_pixels and abs(vehicle_center_y - line_b_y) < line_distance_pixels:
        # Vehicle is between the lines - use INTEGRATION method for accurate distance
        # Integrate the inverse of pixels_per_meter from line_a to line_b
        # distance = integral(dy / pixels_per_meter(y)) from line_a to line_b
        
        # Numerical integration using trapezoidal rule with fine steps
        num_steps = 100
        y_values = np.linspace(line_a_y, line_b_y, num_steps)
        dy = (line_b_y - line_a_y) / (num_steps - 1)
        
        # Calculate distance contribution from each segment
        total_distance = 0.0
        for i in range(len(y_values) - 1):
            y1, y2 = y_values[i], y_values[i+1]
            ppm1 = pixels_per_meter_at_y(y1)
            ppm2 = pixels_per_meter_at_y(y2)
            # Trapezoidal rule: distance += dy * (1/ppm1 + 1/ppm2) / 2
            total_distance += dy * (1/ppm1 + 1/ppm2) / 2
        
        distance_meters = total_distance
        confidence = vehicle_info.get('confidence', 0.7) * 0.9  # High confidence with integration
        method = "integrated_perspective_model"
        
        # Calculate equivalent average pixels per meter for comparison
        avg_ppm = line_distance_pixels / distance_meters
        
        print(f"[PerspectiveCalib] Vehicle between lines - INTEGRATED measurement: {distance_meters:.2f}m")
        print(f"[PerspectiveCalib]   → Line A (y={line_a_y}): {pixels_per_meter_at_y(line_a_y):.1f} px/m")
        print(f"[PerspectiveCalib]   → Vehicle (y={vehicle_center_y:.1f}): {pixels_per_meter_at_vehicle:.1f} px/m")
        print(f"[PerspectiveCalib]   → Line B (y={line_b_y}): {pixels_per_meter_at_y(line_b_y):.1f} px/m")
        print(f"[PerspectiveCalib]   → Avg effective: {avg_ppm:.1f} px/m (ratio: {pixels_per_meter_at_y(line_a_y)/pixels_per_meter_at_y(line_b_y):.2f}x)")
    else:
        # Vehicle is outside - use exponential extrapolation
        # Calculate scale ratio between vehicle position and line midpoint
        line_mid_y = (line_a_y + line_b_y) / 2
        y_diff = vehicle_center_y - line_mid_y
        
        # Extrapolate using exponential model
        scale_at_midpoint = pixels_per_meter_at_vehicle * np.exp(-perspective_decay * y_diff)
        
        # Integrate from line_a to line_b using extrapolated model
        num_steps = 100
        y_values = np.linspace(line_a_y, line_b_y, num_steps)
        dy = (line_b_y - line_a_y) / (num_steps - 1)
        
        total_distance = 0.0
        for i in range(len(y_values) - 1):
            y1, y2 = y_values[i], y_values[i+1]
            # Scale relative to midpoint
            ppm1 = scale_at_midpoint * np.exp(perspective_decay * (y1 - line_mid_y))
            ppm2 = scale_at_midpoint * np.exp(perspective_decay * (y2 - line_mid_y))
            total_distance += dy * (1/ppm1 + 1/ppm2) / 2
        
        distance_meters = total_distance
        confidence = vehicle_info.get('confidence', 0.7) * 0.65  # Lower confidence for extrapolation
        method = "extrapolated_exponential_model"
        print(f"[PerspectiveCalib] Vehicle outside lines - EXTRAPOLATED measurement: {distance_meters:.2f}m")
        print(f"[PerspectiveCalib]   → Extrapolation from y={vehicle_center_y:.1f} to midpoint y={line_mid_y:.1f}")
    
    # Sanity check: distance should be reasonable (typically 5-50 meters)
    if distance_meters < 2 or distance_meters > 100:
        print(f"[PerspectiveCalib] WARNING: Distance {distance_meters:.2f}m seems unrealistic")
        confidence *= 0.5
    
    result = {
        "distance_meters": round(distance_meters, 2),
        "confidence": round(confidence, 2),
        "method": method,
        "vehicle_info": vehicle_info,
        "pixels_per_meter": round(pixels_per_meter_at_vehicle, 2),
        "line_distance_pixels": line_distance_pixels,
        "note": f"Calibrated using {vehicle_type} ({dimension_label}: {real_dimension_m:.1f}m)"
    }
    
    print(f"[PerspectiveCalib] Final result: {result}")
    return result


def find_vehicle_between_lines(frame, line_a_y, line_b_y, tracker_boxes):
    """
    Find a vehicle that is currently between the two calibration lines.
    
    Args:
        frame: Current video frame
        line_a_y: Y-coordinate of line A (top)
        line_b_y: Y-coordinate of line B (bottom)
        tracker_boxes: List of tracked vehicle bounding boxes [[x1,y1,x2,y2], ...]
    
    Returns:
        bbox of the best vehicle for calibration, or None
    """
    if not tracker_boxes or len(tracker_boxes) == 0:
        return None
    
    best_vehicle = None
    best_score = 0
    
    for bbox in tracker_boxes:
        x1, y1, x2, y2 = bbox
        vehicle_center_y = (y1 + y2) / 2
        vehicle_height = y2 - y1
        vehicle_width = x2 - x1
        
        # Check if vehicle center is between lines
        if line_a_y <= vehicle_center_y <= line_b_y:
            # Score based on:
            # 1. How centered between lines (prefer middle)
            line_mid_y = (line_a_y + line_b_y) / 2
            center_score = 1.0 - abs(vehicle_center_y - line_mid_y) / (line_b_y - line_a_y)
            
            # 2. Vehicle size (prefer larger, more visible vehicles)
            size_score = min((vehicle_width * vehicle_height) / (frame.shape[0] * frame.shape[1] * 0.1), 1.0)
            
            # 3. Aspect ratio (prefer side-view vehicles for better length measurement)
            aspect_ratio = vehicle_width / max(vehicle_height, 1)
            aspect_score = min(aspect_ratio / 2.0, 1.0) if aspect_ratio > 1.0 else 0.5
            
            # Combined score
            score = center_score * 0.5 + size_score * 0.3 + aspect_score * 0.2
            
            if score > best_score:
                best_score = score
                best_vehicle = bbox
    
    if best_vehicle:
        print(f"[PerspectiveCalib] Found vehicle between lines with score {best_score:.2f}: {best_vehicle}")
    
    return best_vehicle
