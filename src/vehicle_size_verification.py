"""
Vehicle Size Verification using LLM Vision Analysis

This module uses Gemini Vision to compare vehicle apparent sizes at different Y positions
to validate and improve perspective/distance calibration.

Strategy:
1. Capture screenshots of vehicles at line_A and line_B
2. Ask Gemini to compare apparent sizes
3. Use real-world vehicle dimensions to estimate actual distance
4. Cross-validate with homography results
"""

import base64
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crop_vehicle_region(frame: np.ndarray, bbox: List[float], padding: int = 50) -> np.ndarray:
    """
    Crop vehicle region from frame with padding for context.
    
    Args:
        frame: Full frame image
        bbox: [x1, y1, x2, y2] bounding box
        padding: Extra pixels around bbox
        
    Returns:
        Cropped image region
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = frame.shape[:2]
    
    # Add padding
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)
    
    return frame[y1_pad:y2_pad, x1_pad:x2_pad]


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def compare_vehicle_sizes_with_llm(
    vehicle1_crop: np.ndarray,
    vehicle2_crop: np.ndarray,
    vehicle1_bbox: List[float],
    vehicle2_bbox: List[float],
    vehicle1_y: float,
    vehicle2_y: float,
    vehicle_type: str,
    real_world_length: float,
    api_key: str
) -> Optional[Dict]:
    """
    Use Gemini Vision to compare vehicle apparent sizes and estimate distance.
    
    Args:
        vehicle1_crop: Image of vehicle at line_A (far)
        vehicle2_crop: Image of vehicle at line_B (near)
        vehicle1_bbox: Bbox of vehicle 1 [x1, y1, x2, y2]
        vehicle2_bbox: Bbox of vehicle 2
        vehicle1_y: Y-position of vehicle 1 (line_A)
        vehicle2_y: Y-position of vehicle 2 (line_B)
        vehicle_type: Type of vehicle (sedan, truck, etc.)
        real_world_length: Actual vehicle length in meters
        api_key: Gemini API key
        
    Returns:
        Dict with analysis results or None
    """
    logger.info("[VehicleSizeVerification] Comparing vehicle sizes with LLM...")
    
    try:
        # Encode images
        img1_b64 = encode_image_to_base64(vehicle1_crop)
        img2_b64 = encode_image_to_base64(vehicle2_crop)
        
        # Calculate pixel sizes
        bbox1_width = vehicle1_bbox[2] - vehicle1_bbox[0]
        bbox1_height = vehicle1_bbox[3] - vehicle1_bbox[1]
        bbox2_width = vehicle2_bbox[2] - vehicle2_bbox[0]
        bbox2_height = vehicle2_bbox[3] - vehicle2_bbox[1]
        
        # Determine primary dimension (larger dimension = visible orientation)
        if bbox1_width > bbox1_height:
            dim1_name = "width (side view)"
            dim1_pixels = bbox1_width
        else:
            dim1_name = "height (rear/front view)"
            dim1_pixels = bbox1_height
        
        if bbox2_width > bbox2_height:
            dim2_name = "width (side view)"
            dim2_pixels = bbox2_width
        else:
            dim2_name = "height (rear/front view)"
            dim2_pixels = bbox2_height
        
        # Size ratio
        size_ratio = dim2_pixels / dim1_pixels if dim1_pixels > 0 else 1.0
        
        # Construct prompt
        prompt = f"""Analyze these two images of the SAME {vehicle_type} (real-world length: {real_world_length}m).

IMAGE 1 (FAR - Y position {vehicle1_y:.0f}):
- Vehicle appears {bbox1_width:.0f}px × {bbox1_height:.0f}px
- Primary visible dimension: {dim1_pixels:.0f}px ({dim1_name})

IMAGE 2 (NEAR - Y position {vehicle2_y:.0f}):
- Vehicle appears {bbox2_width:.0f}px × {bbox2_height:.0f}px
- Primary visible dimension: {dim2_pixels:.0f}px ({dim2_name})

SIZE RATIO: Image 2 is {size_ratio:.2f}x larger than Image 1

TASK:
1. Visually confirm both are similar vehicle types
2. Assess if the size ratio ({size_ratio:.2f}x) seems reasonable for perspective difference
3. Estimate the real-world distance between the two positions

For elevated highway cameras (15-20m height):
- 1.5x size ratio ≈ 30-50m depth
- 2.0x size ratio ≈ 80-120m depth
- 2.5x size ratio ≈ 150-200m depth
- 3.0x size ratio ≈ 250-350m depth

Return JSON only:
{{
  "vehicles_match": true/false,
  "size_ratio_valid": true/false,
  "estimated_distance_meters": <number>,
  "confidence": <0.0-1.0>,
  "notes": "<brief explanation>"
}}"""

        # Initialize Gemini
        chat = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=2048,
        )
        
        # Send request with both images
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
            ]
        )
        
        logger.info(f"[VehicleSizeVerification] Sending request (size ratio: {size_ratio:.2f}x)")
        response = chat.invoke([message])
        
        # Parse response
        text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"[VehicleSizeVerification] Raw response: {text}")
        
        # Extract JSON
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            lines = cleaned_text.splitlines()
            cleaned_text = "\n".join(lines[1:])
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Find JSON object
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            jmatch = cleaned_text
        elif '{' in cleaned_text and '}' in cleaned_text:
            jmatch = cleaned_text[cleaned_text.find('{'):cleaned_text.rfind('}')+1]
        else:
            logger.error(f"[VehicleSizeVerification] No JSON found in response")
            return None
        
        result = json.loads(jmatch)
        
        # Add computed data
        result['pixel_size_ratio'] = size_ratio
        result['vehicle1_y'] = vehicle1_y
        result['vehicle2_y'] = vehicle2_y
        result['y_distance_pixels'] = abs(vehicle2_y - vehicle1_y)
        
        logger.info(f"[VehicleSizeVerification] ✅ Analysis complete:")
        logger.info(f"  Vehicles match: {result.get('vehicles_match', 'unknown')}")
        logger.info(f"  Size ratio valid: {result.get('size_ratio_valid', 'unknown')}")
        logger.info(f"  Estimated distance: {result.get('estimated_distance_meters', 'unknown')}m")
        logger.info(f"  Confidence: {result.get('confidence', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"[VehicleSizeVerification] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_vehicles_near_lines(
    vehicles_bboxes: List[List[float]],
    vehicle_types: List[str],
    line_a_y: float,
    line_b_y: float,
    tolerance: int = 100
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find vehicles closest to line_A and line_B.
    
    Args:
        vehicles_bboxes: List of bounding boxes
        vehicle_types: List of vehicle types
        line_a_y: Y position of line A
        line_b_y: Y position of line B
        tolerance: Max distance from line to consider
        
    Returns:
        (vehicle_at_A, vehicle_at_B) or (None, None)
    """
    vehicle_at_a = None
    vehicle_at_b = None
    min_dist_a = float('inf')
    min_dist_b = float('inf')
    
    for bbox, v_type in zip(vehicles_bboxes, vehicle_types):
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Check distance to line_A
        dist_a = abs(y_center - line_a_y)
        if dist_a < tolerance and dist_a < min_dist_a:
            min_dist_a = dist_a
            vehicle_at_a = {
                'bbox': bbox,
                'type': v_type,
                'y_position': y_center
            }
        
        # Check distance to line_B
        dist_b = abs(y_center - line_b_y)
        if dist_b < tolerance and dist_b < min_dist_b:
            min_dist_b = dist_b
            vehicle_at_b = {
                'bbox': bbox,
                'type': v_type,
                'y_position': y_center
            }
    
    return vehicle_at_a, vehicle_at_b


def verify_distance_with_vehicle_comparison(
    frame: np.ndarray,
    vehicles_bboxes: List[List[float]],
    vehicle_types: List[str],
    line_a_y: float,
    line_b_y: float,
    vehicle_dimensions: Dict[str, Dict[str, float]],
    api_key: str,
    homography_distance: Optional[float] = None
) -> Optional[Dict]:
    """
    Main function: Verify distance calibration using LLM vehicle size comparison.
    
    Args:
        frame: Current video frame
        vehicles_bboxes: List of detected vehicle bounding boxes
        vehicle_types: List of vehicle types
        line_a_y: Y position of line A (far)
        line_b_y: Y position of line B (near)
        vehicle_dimensions: Known vehicle dimensions
        api_key: Gemini API key
        homography_distance: Optional homography-computed distance for comparison
        
    Returns:
        Dict with verification results or None
    """
    logger.info("[VehicleSizeVerification] Starting vehicle size verification...")
    
    # Find vehicles near lines
    vehicle_a, vehicle_b = find_vehicles_near_lines(
        vehicles_bboxes, vehicle_types, line_a_y, line_b_y
    )
    
    if vehicle_a is None or vehicle_b is None:
        logger.warning("[VehicleSizeVerification] Could not find vehicles near both lines")
        return None
    
    # Prefer same vehicle type for comparison
    if vehicle_a['type'] != vehicle_b['type']:
        logger.warning(f"[VehicleSizeVerification] Different vehicle types: {vehicle_a['type']} vs {vehicle_b['type']}")
        # Use more common type (sedan) as reference
        vehicle_type = 'sedan' if vehicle_a['type'] == 'sedan' or vehicle_b['type'] == 'sedan' else vehicle_a['type']
    else:
        vehicle_type = vehicle_a['type']
    
    # Get real-world dimensions
    if vehicle_type not in vehicle_dimensions:
        vehicle_type = 'sedan'  # Fallback
    
    real_length = vehicle_dimensions[vehicle_type]['length']
    
    logger.info(f"[VehicleSizeVerification] Found {vehicle_type} at Y={vehicle_a['y_position']:.0f} and Y={vehicle_b['y_position']:.0f}")
    
    # Crop vehicle regions
    crop1 = crop_vehicle_region(frame, vehicle_a['bbox'])
    crop2 = crop_vehicle_region(frame, vehicle_b['bbox'])
    
    # Compare with LLM
    result = compare_vehicle_sizes_with_llm(
        crop1, crop2,
        vehicle_a['bbox'], vehicle_b['bbox'],
        vehicle_a['y_position'], vehicle_b['y_position'],
        vehicle_type, real_length,
        api_key
    )
    
    if result:
        # Add comparison with homography if available
        if homography_distance is not None:
            result['homography_distance'] = homography_distance
            result['llm_distance'] = result.get('estimated_distance_meters', 0)
            
            # Calculate agreement
            if result['llm_distance'] > 0:
                ratio = homography_distance / result['llm_distance']
                result['distance_ratio'] = ratio
                result['distances_agree'] = 0.5 < ratio < 2.0  # Within 2x
                
                logger.info(f"[VehicleSizeVerification] Distance comparison:")
                logger.info(f"  Homography: {homography_distance:.1f}m")
                logger.info(f"  LLM estimate: {result['llm_distance']:.1f}m")
                logger.info(f"  Ratio: {ratio:.2f}x")
                logger.info(f"  Agreement: {'✅ Yes' if result['distances_agree'] else '❌ No'}")
        
        # Compute final calibrated distance (weighted average if both available)
        if homography_distance and result.get('llm_distance', 0) > 0:
            # Weight by confidence
            llm_confidence = result.get('confidence', 0.5)
            homography_weight = 0.6  # Homography is more reliable generally
            llm_weight = 0.4 * llm_confidence
            
            total_weight = homography_weight + llm_weight
            result['calibrated_distance'] = (
                (homography_distance * homography_weight + 
                 result['llm_distance'] * llm_weight) / total_weight
            )
            result['calibration_method'] = 'hybrid_homography_llm'
            logger.info(f"[VehicleSizeVerification] Calibrated distance (hybrid): {result['calibrated_distance']:.1f}m")
        elif result.get('llm_distance', 0) > 0:
            result['calibrated_distance'] = result['llm_distance']
            result['calibration_method'] = 'llm_only'
        elif homography_distance:
            result['calibrated_distance'] = homography_distance
            result['calibration_method'] = 'homography_only'
    
    return result
