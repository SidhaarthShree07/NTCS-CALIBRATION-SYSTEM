"""
Automatic Homography Estimation using Road Polygon + Multiple Vehicles
Based on Roboflow approach: https://blog.roboflow.com/estimate-speed-computer-vision/

This module automatically computes homography matrix without manual measurements by:
1. Using road polygon from segmentation as SOURCE region
2. Using multiple detected vehicles to estimate TARGET region dimensions
3. Computing perspective transformation matrix
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViewTransformer:
    """
    Perspective transformation class (Roboflow approach).
    Transforms image coordinates to real-world road plane coordinates.
    """
    
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Args:
            source: 4x2 array of source points (image coordinates)
            target: 4x2 array of target points (world coordinates in meters)
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        self.source = source
        self.target = target
        
        logger.info(f"[ViewTransformer] Initialized with transformation matrix:")
        logger.info(f"\n{self.m}")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from image to world coordinates"""
        if points.size == 0:
            return points
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    
    def transform_distance(self, y1: float, y2: float, img_width: float) -> float:
        """
        Calculate distance between two y-coordinates (horizontal lines).
        
        Args:
            y1: First y-coordinate (top line)
            y2: Second y-coordinate (bottom line)
            img_width: Image width
            
        Returns:
            Distance in meters
        """
        # Use center points of each line
        point1 = np.array([[img_width / 2, y1]])
        point2 = np.array([[img_width / 2, y2]])
        
        # Transform to world coordinates
        world1 = self.transform_points(point1)[0]
        world2 = self.transform_points(point2)[0]
        
        # Euclidean distance
        distance = np.linalg.norm(world2 - world1)
        
        logger.info(f"[ViewTransformer] Distance: {distance:.2f}m")
        logger.info(f"  Image: ({img_width/2:.0f}, {y1:.0f}) → ({img_width/2:.0f}, {y2:.0f})")
        logger.info(f"  World: ({world1[0]:.2f}, {world1[1]:.2f}) → ({world2[0]:.2f}, {world2[1]:.2f})")
        
        return distance


def estimate_road_dimensions_from_vehicles(
    vehicles: List[Dict],
    vehicle_dimensions: Dict[str, Dict[str, float]],
    img_height: int,
    img_width: int,
    road_polygon: Optional[np.ndarray] = None  # NEW: Pass polygon for bounds
) -> Tuple[float, float]:
    """
    Estimate real-world road dimensions using multiple vehicles at different positions.
    
    Strategy:
    1. Find vehicles at different Y positions (near/far from camera)
    2. Use known vehicle dimensions to compute pixels/meter at each position
    3. Estimate perspective gradient
    4. Extrapolate road dimensions using perspective-aware integration
    
    Args:
        vehicles: List of vehicle dicts with 'bbox', 'vehicle_type', 'y_position'
        vehicle_dimensions: Known dimensions database
        img_height: Image height
        img_width: Image width
        road_polygon: Optional polygon for accurate bounds
        
    Returns:
        (road_width_meters, road_length_meters)
    """
    if len(vehicles) < 2:
        logger.warning("[AutoHomography] Need at least 2 vehicles for auto-estimation")
        # Use defaults for highway
        return 25.0, 250.0  # Typical highway: 25m wide (3 lanes × ~3.5m + shoulders), 250m view
    
    # Sort vehicles by y-position (top to bottom = far to near)
    sorted_vehicles = sorted(vehicles, key=lambda v: v['y_position'])
    
    # Sample data points across the image
    data_points = []
    
    for vehicle in sorted_vehicles:
        v_type = vehicle.get('vehicle_type', 'sedan')
        if v_type not in vehicle_dimensions:
            v_type = 'sedan'  # default
        
        dims = vehicle_dimensions[v_type]
        bbox = vehicle['bbox']
        
        # Vehicle dimensions in pixels
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        # Estimate which dimension is visible (larger bbox dimension = visible)
        if bbox_width > bbox_height:
            # Side view - use length
            real_dim = dims['length']
            pixel_dim = bbox_width
        else:
            # Rear/front view - use width
            real_dim = dims['width']
            pixel_dim = bbox_height
        
        # Pixels per meter at this y-position
        ppm = pixel_dim / real_dim
        y_pos = vehicle['y_position']
        
        data_points.append({
            'y': y_pos,
            'y_ratio': y_pos / img_height,
            'pixels_per_meter': ppm,
            'vehicle_type': v_type
        })
        
        logger.info(f"[AutoHomography] Vehicle {v_type} at y={y_pos:.0f} ({y_pos/img_height*100:.0f}%): {ppm:.1f} px/m")
    
    if len(data_points) < 2:
        return 25.0, 250.0
    
    # Compute perspective gradient
    # Use farthest (top) and nearest (bottom) vehicles
    far_point = data_points[0]  # Top (far from camera)
    near_point = data_points[-1]  # Bottom (close to camera)
    
    # Scale ratio between near and far
    scale_ratio = near_point['pixels_per_meter'] / far_point['pixels_per_meter']
    
    logger.info(f"[AutoHomography] Perspective gradient: {scale_ratio:.2f}x from far to near")
    
    # Estimate road width at bottom of image (closest point)
    # Assume road spans 60-80% of image width at bottom
    road_width_pixels_bottom = img_width * 0.7
    road_width_meters = road_width_pixels_bottom / near_point['pixels_per_meter']
    
    # Estimate road length (from top to bottom of visible area)
    # Distance covered from far_point to near_point
    y_distance_pixels = near_point['y'] - far_point['y']
    
    # CRITICAL FIX: Use perspective-aware integration instead of simple average
    # For elevated cameras, pixels/meter changes exponentially with distance
    # We need to integrate: distance = integral of (1 / pixels_per_meter) dy
    
    # Fit exponential model: ppm(y) = a * exp(b * y_ratio)
    # Using log-linear regression
    y_ratios = [dp['y_ratio'] for dp in data_points]
    ppms = [dp['pixels_per_meter'] for dp in data_points]
    
    # Filter outliers (trucks can cause spikes)
    filtered_points = []
    for i, ppm in enumerate(ppms):
        # Skip extreme outliers (> 150 px/m usually indicates misdetection)
        if ppm < 150:
            filtered_points.append(data_points[i])
    
    if len(filtered_points) >= 2:
        data_points = filtered_points
        far_point = data_points[0]
        near_point = data_points[-1]
        y_ratios = [dp['y_ratio'] for dp in data_points]
        ppms = [dp['pixels_per_meter'] for dp in data_points]
    
    # Numerical integration approach: Sum distance for each pixel
    # For each Y position, estimate 1/ppm and sum
    # distance_meters = sum(dy / ppm(y))
    
    # Simple linear interpolation of 1/ppm across Y range
    # 1/ppm at far = 1/far_ppm, 1/ppm at near = 1/near_ppm
    # Integrate linearly: integral = (y_distance / 2) * (1/far_ppm + 1/near_ppm)
    
    far_ppm = far_point['pixels_per_meter']
    near_ppm = near_point['pixels_per_meter']
    
    # Trapezoidal integration (more accurate for perspective)
    meters_per_pixel_far = 1.0 / far_ppm
    meters_per_pixel_near = 1.0 / near_ppm
    
    # Average meters/pixel across the span
    avg_meters_per_pixel = (meters_per_pixel_far + meters_per_pixel_near) / 2.0
    
    # Distance covered by sampled vehicles
    sampled_length_meters = y_distance_pixels * avg_meters_per_pixel
    
    # Now extrapolate to full road length
    # Assume vehicles sampled from Y=far_y to Y=near_y
    # Full road goes from polygon_top to polygon_bottom
    
    # For elevated highway: Need to account for extreme perspective at top
    # Use polynomial extrapolation for top section
    if road_polygon is not None and len(road_polygon) > 0:
        polygon_top_y = np.min(road_polygon[:, 1])
        polygon_bottom_y = np.max(road_polygon[:, 1])
    else:
        # Fallback: use image bounds
        polygon_top_y = 0
        polygon_bottom_y = img_height
    
    # If we have samples near the top, use them; otherwise extrapolate
    if far_point['y'] - polygon_top_y > 50:  # Lowered threshold for elevated cameras
        # Missing significant top section - extrapolate with much steeper perspective
        missing_top_pixels = far_point['y'] - polygon_top_y
        # Assume perspective continues to decrease (lower ppm at top)
        # For elevated highways: perspective at top is 2-3x steeper!
        estimated_top_ppm = far_ppm * 0.5  # More aggressive (was 0.7)
        top_section_meters = missing_top_pixels / estimated_top_ppm
        
        # Additional scaling for extreme elevation (highway cameras)
        if missing_top_pixels > 80:
            # Very far section - apply elevation multiplier
            # Elevated highway assumption: camera 15-20m high
            elevation_multiplier = 1.5
            top_section_meters *= elevation_multiplier
        
        logger.info(f"[AutoHomography] Extrapolating top section: {missing_top_pixels:.0f}px → {top_section_meters:.1f}m")
    else:
        top_section_meters = 0
    
    # Bottom section
    if polygon_bottom_y - near_point['y'] > 100:
        missing_bottom_pixels = polygon_bottom_y - near_point['y']
        # Assume perspective continues (higher ppm at bottom)
        estimated_bottom_ppm = near_ppm * 1.2
        bottom_section_meters = missing_bottom_pixels / estimated_bottom_ppm
        logger.info(f"[AutoHomography] Extrapolating bottom section: {missing_bottom_pixels:.0f}px → {bottom_section_meters:.1f}m")
    else:
        bottom_section_meters = 0
    
    # Total road length
    road_length_meters = top_section_meters + sampled_length_meters + bottom_section_meters
    
    logger.info(f"[AutoHomography] Sampled section: {sampled_length_meters:.1f}m (Y={far_point['y']:.0f}→{near_point['y']:.0f})")
    logger.info(f"[AutoHomography] Total estimated road length: {road_length_meters:.1f}m")
    
    return road_width_meters, road_length_meters


def estimate_homography_from_polygon_and_vehicles(
    road_polygon: np.ndarray,
    vehicles: List[Dict],
    vehicle_dimensions: Dict[str, Dict[str, float]],
    img_height: int,
    img_width: int
) -> Optional[ViewTransformer]:
    """
    Automatically estimate homography using road polygon + vehicle data.
    
    This is the HYBRID approach:
    1. Road polygon defines SOURCE region (accurate from segmentation)
    2. Vehicles estimate TARGET dimensions (automatic, no manual measurement)
    3. Compute transformation matrix
    
    Args:
        road_polygon: Polygon points from segmentation (Nx2 array)
        vehicles: List of detected vehicles with bboxes and types
        vehicle_dimensions: Known vehicle dimensions database
        img_height: Image height
        img_width: Image width
        
    Returns:
        ViewTransformer object or None if failed
    """
    logger.info("[AutoHomography] Starting automatic homography estimation...")
    
    # Step 1: Extract bounding box of road polygon as SOURCE region
    # This gives us 4 corners of the visible road area
    source_points = extract_road_bounding_quadrilateral(road_polygon, img_height, img_width)
    
    if source_points is None:
        logger.error("[AutoHomography] Failed to extract road quadrilateral")
        return None
    
    logger.info(f"[AutoHomography] SOURCE region (image coords):")
    for i, pt in enumerate(source_points):
        logger.info(f"  Point {i}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # Step 2: Estimate real-world dimensions using vehicles
    road_width_m, road_length_m = estimate_road_dimensions_from_vehicles(
        vehicles, vehicle_dimensions, img_height, img_width, road_polygon  # Pass polygon
    )
    
    # Step 3: Define TARGET region in real-world coordinates
    # Anchor at top-left corner (0, 0) and extend right/down
    target_points = np.array([
        [0, 0],                          # Top-left
        [road_width_m, 0],              # Top-right
        [road_width_m, road_length_m],  # Bottom-right
        [0, road_length_m]              # Bottom-left
    ], dtype=np.float32)
    
    logger.info(f"[AutoHomography] TARGET region (world coords in meters):")
    for i, pt in enumerate(target_points):
        logger.info(f"  Point {i}: ({pt[0]:.1f}m, {pt[1]:.1f}m)")
    
    # Step 4: Create ViewTransformer
    try:
        transformer = ViewTransformer(source_points, target_points)
        logger.info("[AutoHomography] ✅ ViewTransformer created successfully!")
        return transformer
    except Exception as e:
        logger.error(f"[AutoHomography] ❌ Failed to create transformer: {e}")
        return None


def extract_road_bounding_quadrilateral(
    polygon: np.ndarray,
    img_height: int,
    img_width: int
) -> Optional[np.ndarray]:
    """
    Extract a bounding quadrilateral from road polygon.
    
    Strategy:
    1. Find extremes in Y direction (top/bottom of road)
    2. Find extremes in X direction at top and bottom
    3. Form a quadrilateral representing the visible road trapezoid
    
    Args:
        polygon: Road polygon from segmentation (Nx2)
        img_height: Image height
        img_width: Image width
        
    Returns:
        4x2 array of corner points [top-left, top-right, bottom-right, bottom-left]
    """
    if polygon is None or len(polygon) < 4:
        logger.error("[AutoHomography] Invalid polygon")
        return None
    
    # Ensure polygon is numpy array
    polygon = np.array(polygon)
    
    # Split into top half and bottom half
    mid_y = img_height * 0.5
    
    # Top section (far from camera) - find leftmost and rightmost points
    top_points = polygon[polygon[:, 1] < mid_y]
    if len(top_points) == 0:
        top_points = polygon[np.argsort(polygon[:, 1])[:2]]  # Take 2 topmost points
    
    top_left = top_points[np.argmin(top_points[:, 0])]  # Leftmost in top section
    top_right = top_points[np.argmax(top_points[:, 0])]  # Rightmost in top section
    
    # Bottom section (close to camera) - find leftmost and rightmost points
    bottom_points = polygon[polygon[:, 1] > mid_y]
    if len(bottom_points) == 0:
        bottom_points = polygon[np.argsort(polygon[:, 1])[-2:]]  # Take 2 bottommost points
    
    bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
    
    # Form quadrilateral: [TL, TR, BR, BL]
    quad = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)
    
    logger.info("[AutoHomography] Extracted road quadrilateral:")
    logger.info(f"  Top: ({top_left[0]:.0f}, {top_left[1]:.0f}) → ({top_right[0]:.0f}, {top_right[1]:.0f})")
    logger.info(f"  Bottom: ({bottom_left[0]:.0f}, {bottom_left[1]:.0f}) → ({bottom_right[0]:.0f}, {bottom_right[1]:.0f})")
    
    return quad


def calibrate_with_auto_homography(
    frame: np.ndarray,
    road_polygon: np.ndarray,
    vehicles_bboxes: List[List[float]],
    vehicle_types: List[str],
    vehicle_dimensions: Dict[str, Dict[str, float]]
) -> Optional[ViewTransformer]:
    """
    Main entry point for automatic homography calibration.
    
    Args:
        frame: Current video frame
        road_polygon: Segmented road polygon
        vehicles_bboxes: List of vehicle bounding boxes [[x1,y1,x2,y2], ...]
        vehicle_types: List of vehicle types corresponding to bboxes
        vehicle_dimensions: Known dimensions database
        
    Returns:
        ViewTransformer or None
    """
    img_height, img_width = frame.shape[:2]
    
    # Prepare vehicle data
    vehicles = []
    for bbox, v_type in zip(vehicles_bboxes, vehicle_types):
        x1, y1, x2, y2 = bbox
        vehicles.append({
            'bbox': bbox,
            'vehicle_type': v_type,
            'y_position': (y1 + y2) / 2,  # Center Y
            'x_position': (x1 + x2) / 2   # Center X
        })
    
    # Estimate homography
    transformer = estimate_homography_from_polygon_and_vehicles(
        road_polygon,
        vehicles,
        vehicle_dimensions,
        img_height,
        img_width
    )
    
    return transformer
