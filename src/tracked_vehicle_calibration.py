"""
Tracked Vehicle Calibration - Enhanced LLM-First Approach

This module tracks a SINGLE vehicle from line A to line B, captures screenshots
at both positions WITH visible calibration lines, sends to Gemini Vision to
estimate the real-world distance, then uses that distance to configure homography.

Key improvements over random vehicle sampling:
1. Same vehicle tracked = fair size comparison
2. Screenshots include visible calibration lines = spatial context
3. Road lane markings visible = additional distortion reference
4. LLM estimates distance FIRST, then homography aligns to it
5. Homography becomes a "straightening transform" based on LLM's judgment

Workflow:
  Track vehicle A→B → Capture screenshots → LLM estimates distance → Configure homography
"""

import cv2
import numpy as np
import base64
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from auto_homography import ViewTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string for Gemini API."""
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


def draw_calibration_context(frame: np.ndarray, line_a_y: int, line_b_y: int, 
                              vehicle_bbox: List[float], line_label: str,
                              road_polygon: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw calibration lines, vehicle highlight, and trapezoid overlay on frame for LLM context.
    
    Args:
        frame: Original frame
        line_a_y: Y-coordinate of line A (entry/far)
        line_b_y: Y-coordinate of line B (exit/near)
        vehicle_bbox: [x1, y1, x2, y2] of vehicle
        line_label: "LINE A" or "LINE B" to indicate current position
        road_polygon: Optional road polygon to draw cropped trapezoid overlay
    
    Returns:
        Annotated frame with context
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Draw speed zone trapezoid if road polygon provided
    if road_polygon is not None:
        try:
            from calib_server import create_cropped_speed_polygon
            cropped_trap = create_cropped_speed_polygon(road_polygon, line_a_y, line_b_y)
            # Draw trapezoid with semi-transparent overlay
            overlay = annotated.copy()
            cv2.polylines(overlay, [cropped_trap.astype(np.int32)], True, (0, 255, 255), 3)  # Cyan outline
            cv2.fillPoly(overlay, [cropped_trap.astype(np.int32)], (0, 255, 255))  # Cyan fill
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)  # Blend for transparency
            
            # Add trapezoid label
            trap_top = int(np.min(cropped_trap[:, 1]))
            trap_center_x = int(np.mean(cropped_trap[:, 0]))
            cv2.putText(annotated, "SPEED ZONE TRAPEZOID", (trap_center_x - 150, trap_top - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print(f"[TrackedVehicle] Warning: Could not draw trapezoid overlay: {e}")
    
    # Draw both calibration lines (so LLM sees full context)
    cv2.line(annotated, (0, line_a_y), (w, line_a_y), (255, 255, 0), 3)  # Yellow line A
    cv2.line(annotated, (0, line_b_y), (w, line_b_y), (255, 0, 255), 3)  # Magenta line B
    
    # Add line labels
    cv2.putText(annotated, "LINE A (ENTRY/FAR)", (10, line_a_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(annotated, "LINE B (EXIT/NEAR)", (10, line_b_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    
    # Highlight the vehicle with bounding box
    x1, y1, x2, y2 = [int(c) for c in vehicle_bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
    
    # Add current position indicator
    cv2.putText(annotated, f"VEHICLE AT {line_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add pixel distance measurement
    pixel_distance = abs(line_b_y - line_a_y)
    mid_y = (line_a_y + line_b_y) // 2
    cv2.putText(annotated, f"Pixel Distance: {pixel_distance}px", (w - 400, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated


def estimate_distance_from_tracked_vehicle(
    frame_at_line_a: np.ndarray,
    frame_at_line_b: np.ndarray,
    vehicle_bbox_a: List[float],
    vehicle_bbox_b: List[float],
    line_a_y: int,
    line_b_y: int,
    vehicle_type: str,
    gemini_api_key: str,
    road_polygon: Optional[np.ndarray] = None
) -> Dict:
    """
    Estimate real-world distance between lines using LLM vision analysis.
    
    This sends TWO annotated screenshots to Gemini:
    1. Vehicle at line A (with both lines visible)
    2. Vehicle at line B (with both lines visible)
    
    LLM analyzes:
    - Vehicle size change (perspective cue)
    - Road lane markings (distortion reference)
    - Visible calibration lines (spatial context)
    - Camera elevation cues
    
    Args:
        frame_at_line_a: Frame when vehicle crossed line A
        frame_at_line_b: Frame when vehicle crossed line B
        vehicle_bbox_a: [x1, y1, x2, y2] at line A
        vehicle_bbox_b: [x1, y1, x2, y2] at line B
        line_a_y: Y-coordinate of line A
        line_b_y: Y-coordinate of line B
        vehicle_type: Type of vehicle (sedan, truck, bus, motorcycle)
        gemini_api_key: Gemini API key
        road_polygon: Optional road polygon for additional context
    
    Returns:
        Dict with:
        - status: 'success' or 'error'
        - estimated_distance_meters: LLM's distance estimate
        - confidence: LLM's confidence score
        - pixel_distance: Pixel distance between lines
        - vehicle_size_ratio: Apparent size ratio (bbox_b / bbox_a)
        - notes: LLM's reasoning
    """
    logger.info(f"[TrackedCalibration] Estimating distance using tracked {vehicle_type}")
    
    try:
        # Calculate vehicle dimensions in pixels at both positions
        x1_a, y1_a, x2_a, y2_a = vehicle_bbox_a
        x1_b, y1_b, x2_b, y2_b = vehicle_bbox_b
        
        width_a = x2_a - x1_a
        height_a = y2_a - y1_a
        width_b = x2_b - x1_b
        height_b = y2_b - y1_b
        
        # Calculate size ratios (how much bigger vehicle appears at line B vs A)
        width_ratio = width_b / width_a if width_a > 0 else 1.0
        height_ratio = height_b / height_a if height_a > 0 else 1.0
        avg_size_ratio = (width_ratio + height_ratio) / 2
        
        logger.info(f"  Vehicle size ratio: {avg_size_ratio:.2f}x (width: {width_ratio:.2f}, height: {height_ratio:.2f})")
        
        # Pixel distance between lines
        pixel_distance = abs(line_b_y - line_a_y)
        logger.info(f"  Pixel distance: {pixel_distance}px")
        
        # Annotate frames with context (including trapezoid overlay if road polygon provided)
        annotated_a = draw_calibration_context(frame_at_line_a, line_a_y, line_b_y, 
                                                 vehicle_bbox_a, "LINE A", road_polygon)
        annotated_b = draw_calibration_context(frame_at_line_b, line_a_y, line_b_y,
                                                 vehicle_bbox_b, "LINE B", road_polygon)

        # Save annotated images to disk for manual inspection
        out_dir = os.path.join(os.path.dirname(__file__), 'calib')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        a_path = os.path.join(out_dir, f"tracked_vehicle_line_a_{ts}.jpg")
        b_path = os.path.join(out_dir, f"tracked_vehicle_line_b_{ts}.jpg")
        try:
            cv2.imwrite(a_path, annotated_a)
            cv2.imwrite(b_path, annotated_b)
            logger.info(f"  Saved annotated images: A={a_path}, B={b_path}")
        except Exception as save_err:
            logger.warning(f"  Failed to save annotated images: {save_err}")
        
        # Encode images to base64
        img_a_b64 = encode_image_to_base64(annotated_a)
        img_b_b64 = encode_image_to_base64(annotated_b)
        
        logger.info(f"  Encoded images: A={len(img_a_b64)} chars, B={len(img_b_b64)} chars")
        
        # Get image dimensions for context
        img_height, img_width = frame_at_line_a.shape[:2]
        
        # Build concise prompt for Gemini (trimmed to avoid output truncation)
        prompt = f"""You are a computer vision expert. Calculate the REAL-WORLD DISTANCE between two calibration lines in a traffic camera.

IMAGES PROVIDED:
- Image 1: Vehicle at LINE A (yellow line, Y={line_a_y}) - FAR position (smaller)
- Image 2: SAME vehicle at LINE B (magenta line, Y={line_b_y}) - NEAR position (larger)
- The target vehicle has a GREEN BOUNDING BOX in both images. ONLY use this vehicle. Ignore all others.

PIXEL DATA (pre-measured):
- Line A to B pixel distance: {pixel_distance}px
- Image resolution: {img_width} x {img_height}
- Vehicle at LINE A: {width_a:.0f}px wide x {height_a:.0f}px tall
- Vehicle at LINE B: {width_b:.0f}px wide x {height_b:.0f}px tall
- Size ratio (near/far): {avg_size_ratio:.2f}x

STEP 1: Identify the GREEN-BOXED vehicle (category, make/model, real width in meters).
Common widths: Sedan ~1.80m, SUV ~1.90m, Truck ~2.10m, Bus ~2.50m, Motorcycle ~0.80m.

STEP 2: Estimate the camera height in meters. Then compute a CONTINUOUS adjustment factor:
  factor = 0.9 + (estimated_height_meters / 13.0)
  Examples: 5m -> factor 1.28 | 10m -> factor 1.67 | 12m -> factor 1.82 | 18m -> factor 2.28 | 25m -> factor 2.82
  This factor corrects for camera tilt and ground-plane perspective projection.

STEP 3: Calculate distance using perspective geometry (MANDATORY - do NOT use heuristic ratio tables):
  PPM_A = vehicle_pixels_width_A / real_width_meters
  PPM_B = vehicle_pixels_width_B / real_width_meters
  MPP_A = 1/PPM_A
  MPP_B = 1/PPM_B
  scale_diff = MPP_A - MPP_B
  distance = (1 / scale_diff) x factor   (factor from STEP 2)

Example: width=1.80m, 82px at A, 175px at B, camera at 18m:
  factor = 0.9 + 18/13 = 2.28
  PPM_A=82/1.80=45.6, PPM_B=175/1.80=97.2
  MPP_A=0.0219, MPP_B=0.0103, scale_diff=0.0116
  distance = (1/0.0116) x 2.28 = 196.6m

If road markings visible, count dashes (3m dash + 9m gap = 12m/cycle) as sanity check.
No typical-range assumptions. Report computed result even if unusually large or small.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "vehicle_identification": {{
    "category": "<sedan|suv|truck|bus|motorcycle>",
    "specific_model": "<make/model or unknown>",
    "real_length_meters": 0.0,
    "real_width_meters": 0.0,
    "real_height_meters": 0.0,
    "confidence_in_id": 0.0,
    "note": "GREEN-BOXED vehicle only"
  }},
  "camera_analysis": {{
    "estimated_height_meters": 0.0,
    "height_category": "<low|medium|high|very_high>",
    "viewing_angle": "<straight|oblique>",
    "field_of_view": "<wide|normal|telephoto>",
    "perspective_compression": "<weak|moderate|strong|extreme>"
  }},
  "distance_calculation": {{
    "method_a_geometric_meters": 0.0,
    "method_b_road_markings_meters": null,
    "methods_agree": true,
    "final_distance_meters": 0.0,
    "calculation_details": "<brief math summary>"
  }},
  "confidence": 0.0,
  "reasoning": "<2-3 sentences max>",
  "warnings": "<any concerns or none>"
}}"""

        # Log the exact prompt that will be sent to LLM for reproducibility
        logger.info("[TrackedCalibration] Prompt to Gemini (distance):\n" + prompt)
        logger.info(f"[TrackedCalibration] Sending images to Gemini:\n  Image A: {a_path}\n  Image B: {b_path}")

        # Initialize Gemini
        logger.info("[TrackedCalibration] Calling Gemini Vision API...")
        chat = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=gemini_api_key,
            temperature=0.0,  # Deterministic as possible for consistent geometry
            max_output_tokens=8192,  # Large enough to avoid truncation of distance_calculation
        )
        
        # Send both images with prompt
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_a_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b_b64}"}
                }
            ]
        )
        
        response = chat.invoke([human_message])
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"[TrackedCalibration] Gemini response: {response_text}")
        
        # Parse JSON response
        cleaned_text = response_text.strip()
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
        
        # Extract JSON - handle truncated responses robustly
        if '{' in cleaned_text and '}' in cleaned_text:
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}')
            json_str = cleaned_text[json_start:json_end+1]
            
            # Try to parse; if it fails due to truncation, attempt to repair
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"[TrackedCalibration] Initial JSON parse failed: {e}")
                logger.info("[TrackedCalibration] Attempting to repair truncated JSON...")
                
                # Common issue: truncated string in reasoning/warnings field
                # Try to close any open strings and objects
                repaired = json_str
                
                # Count braces to see if we need to close
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                
                # If string is not closed, try to close it
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                
                # Close any open brackets/braces
                repaired += ']' * open_brackets
                repaired += '}' * open_braces
                
                try:
                    result = json.loads(repaired)
                    logger.info("[TrackedCalibration] Successfully repaired truncated JSON")
                except json.JSONDecodeError as e2:
                    logger.error(f"[TrackedCalibration] JSON repair failed: {e2}")
                    raise ValueError(f"Could not parse JSON response (original error: {e}, repair error: {e2})")
        else:
            raise ValueError(f"No JSON found in response: {cleaned_text}")
        
        # Parse structured response
        vehicle_id = result.get('vehicle_identification', {})
        camera_analysis = result.get('camera_analysis', {})
        distance_calc = result.get('distance_calculation', {})
        
        # Extract final distance (try multiple paths for compatibility)
        estimated_distance = None
        if distance_calc and 'final_distance_meters' in distance_calc:
            estimated_distance = float(distance_calc['final_distance_meters'])
        elif 'estimated_distance_meters' in result:
            estimated_distance = float(result['estimated_distance_meters'])
        else:
            raise ValueError("No distance estimate found in response")
        
        confidence = float(result.get('confidence', 0.5))
        
        if estimated_distance <= 0:
            raise ValueError(f"Invalid distance estimate: {estimated_distance}")
        
        # Log detailed results
        logger.info(f"[TrackedCalibration] ✅ LLM Analysis Complete:")
        logger.info(f"  📏 Distance: {estimated_distance:.2f}m (confidence: {confidence:.2f})")
        
        if vehicle_id:
            logger.info(f"  🚗 Vehicle: {vehicle_id.get('category', 'unknown')} - {vehicle_id.get('specific_model', 'unknown')}")
            logger.info(f"      Real dimensions: {vehicle_id.get('real_length_meters', 0):.2f}m × {vehicle_id.get('real_width_meters', 0):.2f}m")
        
        if camera_analysis:
            logger.info(f"  📸 Camera: {camera_analysis.get('height_category', 'unknown')} ({camera_analysis.get('estimated_height_meters', 0):.1f}m)")
            logger.info(f"      Perspective: {camera_analysis.get('perspective_compression', 'unknown')}")
        
        if distance_calc:
            logger.info(f"  🧮 Calculation: {distance_calc.get('calculation_details', 'N/A')}")
            if distance_calc.get('method_b_road_markings_meters'):
                logger.info(f"      Road markings verify: {distance_calc['method_b_road_markings_meters']:.1f}m")
        
        logger.info(f"  💡 Reasoning: {result.get('reasoning', 'N/A')}")
        
        if result.get('warnings') and result['warnings'] != 'none':
            logger.warning(f"  ⚠️  Warnings: {result['warnings']}")
        
        return {
            'status': 'success',
            'estimated_distance_meters': estimated_distance,
            'confidence': confidence,
            'pixel_distance': pixel_distance,
            'vehicle_size_ratio': avg_size_ratio,
            'vehicle_identification': vehicle_id,
            'camera_analysis': camera_analysis,
            'distance_calculation': distance_calc,
            'reasoning': result.get('reasoning', ''),
            'warnings': result.get('warnings', 'none'),
            'perspective_assessment': camera_analysis.get('perspective_compression', 'unknown'),
            'road_marking_observations': distance_calc.get('method_b_road_markings_meters', None),
            'annotated_frame_a': annotated_a,
            'annotated_frame_b': annotated_b,
            'annotated_frame_a_path': a_path,
            'annotated_frame_b_path': b_path
        }
        
    except Exception as e:
        logger.error(f"[TrackedCalibration] Error: {e}")
        import traceback
        traceback.print_exc()
        
        # ---- LOCAL PIXEL-MATH FALLBACK ----
        # If Gemini fails (truncation, network, etc.), estimate distance using
        # the same perspective geometry formula from the prompt.
        try:
            x1_a, y1_a, x2_a, y2_a = vehicle_bbox_a
            x1_b, y1_b, x2_b, y2_b = vehicle_bbox_b
            w_a = x2_a - x1_a
            w_b = x2_b - x1_b
            
            # Typical vehicle widths by type (meters)
            typical_widths = {
                'car': 1.80, 'sedan': 1.80, 'suv': 1.90,
                'truck': 2.10, 'bus': 2.50, 'motorcycle': 0.80,
            }
            real_width = typical_widths.get(vehicle_type, 1.80)
            
            if w_a > 10 and w_b > 10 and w_a != w_b:
                ppm_a = w_a / real_width  # pixels per meter at line A
                ppm_b = w_b / real_width  # pixels per meter at line B
                mpp_a = 1.0 / ppm_a      # meters per pixel at line A
                mpp_b = 1.0 / ppm_b      # meters per pixel at line B
                scale_diff = abs(mpp_a - mpp_b)
                
                if scale_diff > 0.0001:
                    # Estimate camera height from horizon position, use continuous factor
                    img_h = frame_at_line_a.shape[0] if frame_at_line_a is not None else 1080
                    horizon_ratio = min(line_a_y, line_b_y) / img_h
                    # Lower horizon_ratio = higher camera
                    if horizon_ratio < 0.2:
                        est_height = 20.0  # high camera
                    elif horizon_ratio < 0.35:
                        est_height = 12.0  # medium-high
                    elif horizon_ratio < 0.5:
                        est_height = 8.0   # medium
                    else:
                        est_height = 5.0   # low
                    
                    adj = 0.9 + (est_height / 13.0)  # continuous factor
                    
                    fallback_dist = (1.0 / scale_diff) * adj
                    
                    # Sanity: must be between 5m and 500m
                    if 5.0 <= fallback_dist <= 500.0:
                        logger.warning(f"[TrackedCalibration] Using LOCAL FALLBACK distance: {fallback_dist:.1f}m")
                        return {
                            'status': 'success',
                            'estimated_distance_meters': round(fallback_dist, 1),
                            'confidence': 0.45,  # lower confidence for fallback
                            'pixel_distance': abs(line_b_y - line_a_y),
                            'vehicle_size_ratio': (w_b / w_a) if w_a > 0 else 1.0,
                            'vehicle_identification': {'category': vehicle_type, 'note': 'fallback estimation'},
                            'camera_analysis': {},
                            'distance_calculation': {'method': 'local_pixel_math_fallback', 'final_distance_meters': round(fallback_dist, 1)},
                            'reasoning': f'Gemini failed; used local perspective math with {vehicle_type} width={real_width}m, adj={adj}',
                            'warnings': f'LLM failed ({e}), used pixel-math fallback'
                        }
                    else:
                        logger.warning(f"[TrackedCalibration] Fallback distance {fallback_dist:.1f}m outside sanity range [5-500m]")
            logger.warning("[TrackedCalibration] Local fallback could not compute valid distance")
        except Exception as fb_err:
            logger.error(f"[TrackedCalibration] Local fallback also failed: {fb_err}")
        
        return {
            'status': 'error',
            'message': str(e),
            'estimated_distance_meters': None
        }


def configure_homography_from_llm_distance(
    road_polygon: np.ndarray,
    llm_distance_meters: float,
    line_a_y: int,
    line_b_y: int,
    img_width: int,
    img_height: int,
    road_width_estimate_meters: float = 10.0
) -> Optional['ViewTransformer']:
    """
    Configure homography transformation using LLM's distance estimate.
    
    This creates a perspective transformation that "straightens" the road
    based on the LLM's judgment of real-world distance.
    
    Args:
        road_polygon: Road polygon points
        llm_distance_meters: Distance estimated by LLM
        line_a_y: Y-coordinate of line A
        line_b_y: Y-coordinate of line B
        img_width: Image width
        img_height: Image height
        road_width_estimate_meters: Estimated road width (default 10m for highway)
    
    Returns:
        ViewTransformer object configured with LLM-aligned homography
    """
    logger.info("[TrackedCalibration] Configuring homography from LLM distance...")
    
    try:
        from auto_homography import ViewTransformer
        
        # Extract road polygon bounds
        polygon_top_y = int(np.min(road_polygon[:, 1]))
        polygon_bottom_y = int(np.max(road_polygon[:, 1]))
        polygon_left_x = int(np.min(road_polygon[:, 0]))
        polygon_right_x = int(np.max(road_polygon[:, 0]))
        
        logger.info(f"  Road polygon: X={polygon_left_x}-{polygon_right_x}, Y={polygon_top_y}-{polygon_bottom_y}")
        
        # Calculate the proportion of road covered by calibration lines
        line_a_offset_from_top = line_a_y - polygon_top_y
        line_b_offset_from_top = line_b_y - polygon_top_y
        total_polygon_height = polygon_bottom_y - polygon_top_y
        
        # LLM measured distance between lines
        # Extrapolate to full road depth
        line_section_height = line_b_y - line_a_y
        
        if line_section_height <= 0:
            raise ValueError("Invalid line positions")
        
        # Estimate full road depth (extrapolate from LLM measurement)
        # Proportion of road depth covered by lines
        line_section_proportion = line_section_height / total_polygon_height
        
        # If lines cover X% of image, and LLM says that's Y meters,
        # then full road depth = Y / X
        estimated_full_depth = llm_distance_meters / line_section_proportion
        
        logger.info(f"  LLM distance between lines: {llm_distance_meters:.2f}m")
        logger.info(f"  Lines cover {line_section_proportion*100:.1f}% of polygon height")
        logger.info(f"  Extrapolated full road depth: {estimated_full_depth:.2f}m")
        
        # Define SOURCE region (image coordinates - use full polygon)
        SOURCE = np.array([
            [polygon_left_x, polygon_top_y],      # Top-left
            [polygon_right_x, polygon_top_y],     # Top-right
            [polygon_right_x, polygon_bottom_y],  # Bottom-right
            [polygon_left_x, polygon_bottom_y]    # Bottom-left
        ], dtype=np.float32)
        
        # Define TARGET region (real-world meters)
        TARGET = np.array([
            [0, 0],                                    # Top-left (origin)
            [road_width_estimate_meters, 0],          # Top-right
            [road_width_estimate_meters, estimated_full_depth],  # Bottom-right
            [0, estimated_full_depth]                 # Bottom-left
        ], dtype=np.float32)
        
        logger.info(f"  SOURCE region: {SOURCE.tolist()}")
        logger.info(f"  TARGET region: {TARGET.tolist()}")
        
        # Create ViewTransformer
        transformer = ViewTransformer(source=SOURCE, target=TARGET)
        
        logger.info("[TrackedCalibration] ✅ Homography configured successfully!")
        logger.info(f"  Transformation matrix: {transformer.m.tolist()}")
        
        return transformer
        
    except Exception as e:
        logger.error(f"[TrackedCalibration] Error configuring homography: {e}")
        import traceback
        traceback.print_exc()
        return None
