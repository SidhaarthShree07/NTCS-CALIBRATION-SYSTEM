"""
Violation Recording System
- Records EXACTLY 5-second video segments (125 frames at 25 FPS)
- Maintains rolling buffer of last 6 segments (30 seconds total)
- Records ANNOTATED frames with speed overlays, bounding boxes, and labels
- Detects speed violations
- Waits for vehicle to reach middle line before capturing screenshot (240 frames max)
- License plate OCR using Gemini Vision API
- Confidence-based filtering: <90% (reject), 90-94%, 95%+
- Uploads evidence to Azure Blob Storage
- Sends violation events to backend API
"""

import cv2
import os
import time
import threading
import uuid
from datetime import datetime, timezone, timedelta
from collections import deque
from pathlib import Path
import json
import hashlib
import hmac
import numpy as np
import re
import base64
from queue import Queue, Empty
from threading import Lock
import database as db  # Local SQLite database
import subprocess
import shlex

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"[ViolationRecorder] Loaded environment from: {env_path}")
except ImportError:
    print("[ViolationRecorder] python-dotenv not installed. Using system environment variables.")
    print("[ViolationRecorder] Install with: pip install python-dotenv")

# Azure Blob Storage imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    AZURE_AVAILABLE = True
except ImportError:
    print("[ViolationRecorder] Warning: azure-storage-blob not installed. Run: pip install azure-storage-blob")
    AZURE_AVAILABLE = False

# Gemini imports for OCR
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("[ViolationRecorder] Gemini Vision API available for OCR")
except ImportError:
    GEMINI_AVAILABLE = False
    print("[ViolationRecorder] Warning: Gemini not available. Install: pip install langchain-google-genai google-generativeai")

class ViolationRecorder:
    def __init__(self, camera_id, speed_limit=75, line_a_y=None, line_b_y=None):
        """
        Initialize violation recorder.
        
        Args:
            camera_id: Unique camera identifier (e.g., "CAM-CHD-042")
            speed_limit: Speed limit in km/h
            line_a_y: Y coordinate of line A (for middle line calculation)
            line_b_y: Y coordinate of line B (for middle line calculation)
        """
        self.camera_id = camera_id
        self.speed_limit = speed_limit
        
        # Middle line for screenshot trigger
        if line_a_y is not None and line_b_y is not None:
            self.middle_line_y = int((line_a_y + line_b_y) / 2)
            self.line_a_y = line_a_y
            self.line_b_y = line_b_y
            print(f"[ViolationRecorder] Middle line Y: {self.middle_line_y} (between {line_a_y} and {line_b_y})")
        else:
            self.middle_line_y = None
            self.line_a_y = None
            self.line_b_y = None
            print(f"[ViolationRecorder] Warning: No calibration lines provided, middle line disabled")
        
        # Recording configuration
        self.segment_duration = 10  # seconds
        self.max_segments = 3  # Keep last 3 segments (30 seconds total)
        self.fps = 25  # Frames per second
        self.frames_per_segment = self.fps * self.segment_duration  # Exactly 250 frames = 10 seconds
        
        # Violation clip configuration (for trimmed clips sent to Azure)
        self.violation_clip_duration = 5  # seconds - duration of trimmed clip around violation
        
        # Storage paths
        self.proof_dir = Path("temp/proof")
        self.proof_dir.mkdir(parents=True, exist_ok=True)
        
        # Rolling buffer of video segments
        self.video_segments = deque(maxlen=self.max_segments)
        self.segment_lock = threading.Lock()
        
        # Current recording state
        self.current_writer = None
        self.current_segment_path = None  # final path (visible after move)
        self.current_segment_tmp = None   # temp path used while ffmpeg writes
        self.segment_start_time = None
        self.segment_frame_count = 0  # Count frames for exact 5-second duration
        self.frame_buffer = []
        
        # Azure configuration (from environment or config)
        self.azure_connection_string = os.getenv(
            'AZURE_STORAGE_CONNECTION_STRING',
            ''  # Set this in environment
        )
        self.azure_container_name = os.getenv('AZURE_CONTAINER_NAME', 'traffic-violations')
        
        # Backend API configuration (now uses local SQLite)
        self.hmac_secret = os.getenv('HMAC_SECRET', 'your-secret-key-here')  # Change this!
        
        # Violation tracking
        self.detected_violations = set()  # Track by vehicle ID to avoid duplicates
        self.violation_cooldown = {}  # vehicle_id -> timestamp
        self.cooldown_seconds = 30  # Don't report same vehicle twice within 30 seconds
        
        # Vehicle tracking for middle line waiting
        self.waiting_vehicles = {}  # vehicle_id -> {'speed', 'class', 'crossed_middle': False, 'bbox', 'detection_time'}
        
        # File tracking for cleanup after successful upload
        self.violation_files = {}  # vehicle_id -> [list of file paths]
        
        # Violation processing queue (graceful parallel processing)
        self.violation_queue = Queue(maxsize=100)  # Max 100 pending violations
        self.queue_lock = Lock()
        self.processing_workers = 3  # Number of parallel worker threads
        self.workers_running = True
        self.worker_threads = []
        
        # Start violation processing workers
        for i in range(self.processing_workers):
            worker = threading.Thread(target=self._violation_worker, args=(i,), daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        print(f"[ViolationRecorder] Started {self.processing_workers} violation processing workers")
        
        # Violation tracking log file
        self.log_file = self.proof_dir / f"violations_log_{camera_id}.txt"
        self._init_log_file()
        
        # Initialize Gemini for OCR
        self.gemini_ocr = None
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_ocr = genai.GenerativeModel('gemini-3-flash-preview')
                    print("[ViolationRecorder] ✅ Gemini Vision API initialized for OCR")
                else:
                    print("[ViolationRecorder] ⚠️ GEMINI_API_KEY not found in environment")
            except Exception as e:
                print(f"[ViolationRecorder] ⚠️ Gemini initialization failed: {e}")
                self.gemini_ocr = None
        
        # Start background cleanup thread
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_files_loop, daemon=True)
        self.cleanup_thread.start()
        
        print(f"[ViolationRecorder] Initialized for camera {camera_id}")
        print(f"[ViolationRecorder] Speed limit: {speed_limit} km/h")
        print(f"[ViolationRecorder] Proof directory: {self.proof_dir}")
        print(f"[ViolationRecorder] Azure available: {AZURE_AVAILABLE}")
        print(f"[ViolationRecorder] Gemini OCR available: {GEMINI_AVAILABLE and self.gemini_ocr is not None}")
        print(f"[ViolationRecorder] 🧹 Background cleanup thread started (10min threshold)")
    
    def _init_log_file(self):
        """Initialize violation tracking log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Violation Recording Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Camera: {self.camera_id} | Speed Limit: {self.speed_limit} km/h\n")
                f.write(f"{'='*80}\n\n")
            print(f"[ViolationRecorder] 📝 Logging violations to: {self.log_file}")
        except Exception as e:
            print(f"[ViolationRecorder] ⚠️ Could not initialize log file: {e}")

    # --- FFMPEG helper writer (reliable H.264 output using system ffmpeg) ---
    class _FfmpegWriter:
        def __init__(self, tmp_path, width, height, fps=25, crf=23, preset='veryfast'):
            self.tmp_path = str(tmp_path)
            # ffmpeg reads raw BGR frames from stdin
            cmd = (
                f"ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
                f"-c:v libx264 -preset {preset} -crf {crf} -pix_fmt yuv420p -movflags +faststart -f mp4 {shlex.quote(self.tmp_path)}"
            )
            self.proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)

        def write(self, frame: np.ndarray):
            # frame must be BGR uint8
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    pass

        def release(self):
            if self.proc:
                try:
                    if self.proc.stdin:
                        self.proc.stdin.close()
                except Exception:
                    pass
                self.proc.wait()
                self.proc = None

    def _ffmpeg_remux(self, src_path):
        """Remux MP4 to ensure moov atom and faststart (in-place replace)."""
        try:
            src = str(src_path)
            out = src + '.remux.mp4'
            cmd = f"ffmpeg -y -i {shlex.quote(src)} -c copy -movflags +faststart {shlex.quote(out)}"
            subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(out, src)
            return True
        except Exception as e:
            print(f"[ViolationRecorder] ⚠️ Remux failed for {src_path}: {e}")
            return False

    
    def _log_violation(self, vehicle_id, status, details=""):
        """Log violation status to file."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Vehicle {vehicle_id:3d} | {status:20s} | {details}\n")
        except Exception as e:
            print(f"[ViolationRecorder] ⚠️ Log write error: {e}")
    
    def _violation_worker(self, worker_id):
        """Worker thread that processes violations from queue."""
        print(f"[ViolationWorker-{worker_id}] Started")
        
        while self.workers_running:
            try:
                # Get violation from queue (timeout 1 second)
                violation_data = self.violation_queue.get(timeout=1)
                
                if violation_data is None:  # Poison pill
                    break
                
                vehicle_id = violation_data['vehicle_id']
                print(f"[ViolationWorker-{worker_id}] Processing vehicle {vehicle_id} (Queue size: {self.violation_queue.qsize()})")
                self._log_violation(vehicle_id, "QUEUE_PICKED", f"Worker {worker_id}")
                
                # Process the violation
                self._process_violation(
                    violation_data['vehicle_id'],
                    violation_data['vehicle_class'],
                    violation_data['speed_kmh'],
                    violation_data['plate_text'],
                    violation_data['plate_confidence'],
                    violation_data['violation_time'],
                    violation_data.get('frame'),
                    violation_data.get('bbox')
                )
                
                self.violation_queue.task_done()
                
            except Empty:
                continue  # No items in queue, keep waiting
            except Exception as e:
                print(f"[ViolationWorker-{worker_id}] Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[ViolationWorker-{worker_id}] Stopped")
    
    def add_frame(self, frame, timestamp=None, vehicle_data=None):
        """
        Add a frame to the current recording segment.
        
        Args:
            frame: OpenCV frame (BGR) - should already have overlays
            timestamp: Frame timestamp (defaults to current time)
            vehicle_data: Optional dict with vehicle info for this frame
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize new segment if needed (check frame count for exact 5-second duration)
        if self.current_writer is None or self.segment_frame_count >= self.frames_per_segment:
            self._start_new_segment(frame, timestamp)
        
        # Write frame to current segment
        if self.current_writer is not None:
            self.current_writer.write(frame)
            self.segment_frame_count += 1
            self.frame_buffer.append({
                'timestamp': timestamp,
                'frame': frame.copy(),
                'vehicle_data': vehicle_data  # Store vehicle info with frame
            })
    
    def _start_new_segment(self, frame, timestamp):
        """Start a new 5-second recording segment."""
        # Close previous segment
        if self.current_writer is not None:
            # Release ffmpeg writer which finalizes tmp file
            try:
                self.current_writer.release()
            except Exception as e:
                print(f"[ViolationRecorder] Warning releasing writer: {e}")

            # Atomically move tmp -> final if exists
            try:
                if self.current_segment_tmp and os.path.exists(self.current_segment_tmp):
                    os.replace(self.current_segment_tmp, self.current_segment_path)
            except Exception as e:
                print(f"[ViolationRecorder] Warning moving tmp to final: {e}")

            # Remux to ensure moov atom and faststart (best-effort)
            try:
                if os.path.exists(self.current_segment_path):
                    self._ffmpeg_remux(self.current_segment_path)
            except Exception:
                pass

            # Add completed segment to rolling buffer
            with self.segment_lock:
                segment_info = {
                    'path': self.current_segment_path,
                    'start_time': self.segment_start_time,
                    'end_time': timestamp,
                    'frame_count': self.segment_frame_count,
                    'duration': self.segment_frame_count / self.fps,
                    'frames': self.frame_buffer.copy()
                }
                self.video_segments.append(segment_info)

                print(f"[ViolationRecorder] Completed segment: {self.segment_frame_count} frames ({self.segment_frame_count / self.fps:.1f}s)")

                # Delete oldest segment file if buffer is full
                if len(self.video_segments) > self.max_segments:
                    old_segment = self.video_segments[0]
                    try:
                        if os.path.exists(old_segment['path']):
                            os.remove(old_segment['path'])
                            print(f"[ViolationRecorder] Deleted old segment: {old_segment['path']}")
                    except Exception as e:
                        print(f"[ViolationRecorder] Error deleting old segment: {e}")
        
        # Create new segment (write to a .tmp path first for atomic move)
        segment_id = uuid.uuid4().hex[:8]
        final_path = self.proof_dir / f"segment_{self.camera_id}_{segment_id}.mp4"
        tmp_path = str(final_path) + '.tmp'
        self.current_segment_tmp = tmp_path
        self.current_segment_path = str(final_path)
        self.segment_start_time = timestamp
        self.segment_frame_count = 0
        self.frame_buffer = []

        # Initialize ffmpeg writer which writes H.264 into tmp path
        height, width = frame.shape[:2]
        try:
            self.current_writer = self._FfmpegWriter(tmp_path, width, height, fps=self.fps)
            print(f"[ViolationRecorder] ✅ Using ffmpeg H.264 writer for streaming support (tmp: {tmp_path})")
        except Exception as e:
            print(f"[ViolationRecorder] ⚠️ ffmpeg writer failed: {e}")
            # Fallback to OpenCV mp4v writer (best-effort)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.current_writer = cv2.VideoWriter(
                str(self.current_segment_path),
                fourcc,
                self.fps,
                (width, height)
            )
            self.current_segment_tmp = None

        print(f"[ViolationRecorder] Started new segment: {self.current_segment_path}")
    
    def check_violation(self, vehicle_id, vehicle_class, speed_kmh, bbox=None, frame=None, plate_text="", plate_confidence=0.0, frame_idx=0):
        """
        Check if vehicle is violating speed limit and track for middle line crossing.
        
        Args:
            vehicle_id: Unique vehicle tracking ID
            vehicle_class: Vehicle class (e.g., "car", "truck", "motorcycle")
            speed_kmh: Measured speed in km/h
            bbox: Bounding box (x1, y1, x2, y2) for vehicle location
            frame: Current frame for capturing screenshot
            plate_text: License plate text (if detected)
            plate_confidence: OCR confidence (0-1)
            frame_idx: Current frame index for timeout calculation (default: 0)
        
        Returns:
            bool: True if violation was recorded, False otherwise
        """
        # Check if speeding
        if speed_kmh <= self.speed_limit:
            # Remove from waiting vehicles if no longer speeding
            if vehicle_id in self.waiting_vehicles:
                del self.waiting_vehicles[vehicle_id]
            return False
        
        # Check cooldown to avoid duplicate reports
        current_time = time.time()
        if vehicle_id in self.violation_cooldown:
            last_report = self.violation_cooldown[vehicle_id]
            if (current_time - last_report) < self.cooldown_seconds:
                return False  # Too soon, skip
        
        # If no middle line configured, process immediately (legacy behavior)
        if self.middle_line_y is None:
            print(f"[ViolationRecorder] 🚨 VIOLATION DETECTED (no middle line)!")
            print(f"[ViolationRecorder]   Vehicle ID: {vehicle_id}")
            print(f"[ViolationRecorder]   Speed: {speed_kmh:.1f} km/h (Limit: {self.speed_limit} km/h)")
            
            # Update cooldown
            self.violation_cooldown[vehicle_id] = current_time
            
            # Process violation immediately
            threading.Thread(
                target=self._process_violation,
                args=(vehicle_id, vehicle_class, speed_kmh, plate_text, plate_confidence, current_time, None),
                daemon=True
            ).start()
            
            return True
        
        # Middle line logic: wait for vehicle to reach middle line
        vehicle_center_y = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            vehicle_center_y = int((y1 + y2) / 2)
        
        # Check if this vehicle already has a violation recorded (duplicate prevention)
        if vehicle_id in self.violation_files or vehicle_id in [v for v in self.waiting_vehicles if self.waiting_vehicles[v].get('crossed_middle')]:
            # This vehicle already triggered a violation, ignore duplicate
            return False
        
        # Track vehicle if not already tracked
        if vehicle_id not in self.waiting_vehicles:
            self.waiting_vehicles[vehicle_id] = {
                'speed': speed_kmh,
                'class': vehicle_class,
                'crossed_middle': False,
                'bbox': bbox,
                'detection_time': current_time,
                'detection_frame': frame_idx,  # Track frame number instead of time
                'before_middle': vehicle_center_y < self.middle_line_y if vehicle_center_y else None,
                'plate_text': plate_text,  # Store initial plate detection
                'plate_confidence': plate_confidence,
                'initial_frame': frame.copy() if frame is not None else None,  # Store initial violation frame
                'initial_bbox': bbox,  # Store initial bbox
                'middle_line_frame': None,  # Will store frame when vehicle crosses middle line
                'middle_line_bbox': None   # Will store bbox at middle line
            }
            print(f"[ViolationRecorder] 🚨 VIOLATION DETECTED - Waiting for middle line")
            print(f"[ViolationRecorder]   Vehicle ID: {vehicle_id}")
            print(f"[ViolationRecorder]   Speed: {speed_kmh:.1f} km/h (Limit: {self.speed_limit} km/h)")
            print(f"[ViolationRecorder]   Vehicle Y: {vehicle_center_y}, Middle line: {self.middle_line_y}")
            print(f"[ViolationRecorder]   Frame: {frame_idx}, will wait up to 240 frames")
            print(f"[ViolationRecorder]   Initial frame saved for 240-frame fallback")
            return False
        
        # Update vehicle data
        vehicle_data = self.waiting_vehicles[vehicle_id]
        vehicle_data['speed'] = speed_kmh
        vehicle_data['bbox'] = bbox
        
        # Check if vehicle crossed middle line
        if vehicle_center_y is not None:
            was_before = vehicle_data.get('before_middle', None)
            is_after = vehicle_center_y >= self.middle_line_y
            
            # Detected before middle line, now crossing or after
            if was_before and is_after and not vehicle_data['crossed_middle']:
                print(f"[ViolationRecorder] ✅ Vehicle {vehicle_id} crossed middle line!")
                print(f"[ViolationRecorder]   Taking screenshot at Y={vehicle_center_y} (Middle: {self.middle_line_y})")
                print(f"[ViolationRecorder] 📤 Queueing violation for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
                
                vehicle_data['crossed_middle'] = True
                vehicle_data['middle_line_frame'] = frame.copy() if frame is not None else None  # Save middle line frame
                vehicle_data['middle_line_bbox'] = bbox  # Save bbox at middle line
                
                # Update cooldown
                self.violation_cooldown[vehicle_id] = current_time
                
                # Add to processing queue instead of spawning thread
                try:
                    self.violation_queue.put({
                        'vehicle_id': vehicle_id,
                        'vehicle_class': vehicle_class,
                        'speed_kmh': speed_kmh,
                        'plate_text': plate_text,
                        'plate_confidence': plate_confidence,
                        'violation_time': current_time,
                        'frame': vehicle_data['middle_line_frame'],  # Use middle line frame, NOT violation frame
                        'bbox': vehicle_data['middle_line_bbox']  # Use middle line bbox
                    }, timeout=5)
                    self._log_violation(vehicle_id, "QUEUED", f"Speed: {speed_kmh:.1f} km/h, Middle line crossed")
                    print(f"[ViolationRecorder] ✅ Violation queued (Queue size: {self.violation_queue.qsize()})")
                except Exception as e:
                    print(f"[ViolationRecorder] ❌ Queue full! Could not queue violation: {e}")
                    self._log_violation(vehicle_id, "QUEUE_FULL", f"Speed: {speed_kmh:.1f} km/h")
                
                # Clean up
                del self.waiting_vehicles[vehicle_id]
                
                return True
            
            # Detected after middle line (violation happened after line)
            elif not was_before and is_after and not vehicle_data['crossed_middle']:
                print(f"[ViolationRecorder] ✅ Vehicle {vehicle_id} detected after middle line!")
                print(f"[ViolationRecorder]   Taking screenshot at Y={vehicle_center_y} (Middle: {self.middle_line_y})")
                print(f"[ViolationRecorder] 📤 Queueing violation for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
                
                vehicle_data['crossed_middle'] = True
                vehicle_data['middle_line_frame'] = frame.copy() if frame is not None else None  # Save current frame as middle line frame
                vehicle_data['middle_line_bbox'] = bbox  # Save current bbox
                
                # Update cooldown
                self.violation_cooldown[vehicle_id] = current_time
                
                # Add to processing queue
                try:
                    self.violation_queue.put({
                        'vehicle_id': vehicle_id,
                        'vehicle_class': vehicle_class,
                        'speed_kmh': speed_kmh,
                        'plate_text': plate_text,
                        'plate_confidence': plate_confidence,
                        'violation_time': current_time,
                        'frame': vehicle_data['middle_line_frame'],  # Use current frame as middle line frame
                        'bbox': vehicle_data['middle_line_bbox']  # Use current bbox
                    }, timeout=5)
                    self._log_violation(vehicle_id, "QUEUED", f"Speed: {speed_kmh:.1f} km/h, After middle line")
                    print(f"[ViolationRecorder] ✅ Violation queued (Queue size: {self.violation_queue.qsize()})")
                except Exception as e:
                    print(f"[ViolationRecorder] ❌ Queue full! Could not queue violation: {e}")
                    self._log_violation(vehicle_id, "QUEUE_FULL", f"Speed: {speed_kmh:.1f} km/h")
                
                # Clean up
                del self.waiting_vehicles[vehicle_id]
                
                return True
        
        # FALLBACK: After 240 frames (~9.6 sec at 25fps), use initial violation frame if vehicle didn't cross middle line
        frames_waited = frame_idx - vehicle_data.get('detection_frame', frame_idx)
        if frames_waited > 240:
            print(f"[ViolationRecorder] ⏱️ Vehicle {vehicle_id} timeout after {frames_waited} frames (didn't cross middle line)")
            print(f"[ViolationRecorder] 🔄 FALLBACK: Using initial violation frame for OCR")
            
            # Get stored initial data
            initial_frame = vehicle_data.get('initial_frame')
            initial_bbox = vehicle_data.get('initial_bbox')
            initial_plate = vehicle_data.get('plate_text')
            initial_confidence = vehicle_data.get('plate_confidence')
            
            if initial_frame is not None and initial_bbox is not None:
                print(f"[ViolationRecorder]   Using frame from {vehicle_data['detection_time']}")
                print(f"[ViolationRecorder]   Initial plate: {initial_plate} ({initial_confidence*100:.1f}% confidence)")
                print(f"[ViolationRecorder] 📤 Queueing fallback violation for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
                
                # Update cooldown
                self.violation_cooldown[vehicle_id] = current_time
                
                # Add to processing queue
                try:
                    self.violation_queue.put({
                        'vehicle_id': vehicle_id,
                        'vehicle_class': vehicle_data['class'],
                        'speed_kmh': vehicle_data['speed'],
                        'plate_text': initial_plate,
                        'plate_confidence': initial_confidence,
                        'violation_time': vehicle_data['detection_time'],
                        'frame': initial_frame,
                        'bbox': initial_bbox
                    }, timeout=5)
                    self._log_violation(vehicle_id, "QUEUED_FALLBACK", f"Speed: {vehicle_data['speed']:.1f} km/h, 240 frames timeout")
                    print(f"[ViolationRecorder] ✅ Fallback violation queued (Queue size: {self.violation_queue.qsize()})")
                except Exception as e:
                    print(f"[ViolationRecorder] ❌ Queue full! Could not queue fallback: {e}")
                    self._log_violation(vehicle_id, "QUEUE_FULL_FALLBACK", f"Speed: {vehicle_data['speed']:.1f} km/h")
            else:
                print(f"[ViolationRecorder]   ❌ No initial frame saved, cannot process violation")
                self._log_violation(vehicle_id, "FALLBACK_FAILED", "No initial frame")
            
            # Clean up
            del self.waiting_vehicles[vehicle_id]
        
        return False
    
    def _process_violation(self, vehicle_id, vehicle_class, speed_kmh, plate_text, plate_confidence, violation_time, frame=None, bbox=None):
        """Process violation: capture evidence, OCR license plate, upload, and send to API."""
        try:
            print(f"[ViolationRecorder] ⏱️ STARTED processing violation for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            print(f"[ViolationRecorder]   Vehicle: {vehicle_id}, Speed: {speed_kmh} km/h, Plate: {plate_text} ({plate_confidence*100:.1f}%)")
            
            # 1. Find the relevant video segment
            video_path = self._get_violation_video(violation_time)
            if not video_path:
                print(f"[ViolationRecorder] No video segment found for violation")
                return
            
            # Verify video file is accessible
            if not os.path.exists(video_path):
                print(f"[ViolationRecorder] ⚠️ Video file does not exist: {video_path}")
            else:
                file_size = os.path.getsize(video_path)
                print(f"[ViolationRecorder] Video file found: {video_path} ({file_size} bytes)")
                
                # Try to verify it's a valid video
                try:
                    import cv2
                    test_cap = cv2.VideoCapture(str(video_path))
                    if test_cap.isOpened():
                        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(f"[ViolationRecorder] Video is readable: {frame_count} frames")
                        test_cap.release()
                    else:
                        print(f"[ViolationRecorder] ⚠️ Video file cannot be opened (might still be writing)")
                except Exception as e:
                    print(f"[ViolationRecorder] ⚠️ Could not verify video: {e}")
            
            # 2. Capture screenshot from violation moment with overlay
            # If frame is provided (from middle line crossing), use it directly
            if frame is not None and bbox is not None:
                print(f"[ViolationRecorder] Using provided frame for screenshot (middle line trigger)")
                screenshot_path = self._capture_screenshot_from_frame(frame, bbox, vehicle_id, vehicle_class, speed_kmh, violation_time)
            else:
                screenshot_path = self._capture_violation_screenshot(violation_time, vehicle_id, vehicle_class, speed_kmh)
            
            if not screenshot_path:
                print(f"[ViolationRecorder] Failed to capture screenshot")
                return
            
            # 3. Run OCR on cropped vehicle image
            ocr_result = self._run_ocr_on_vehicle(screenshot_path, bbox, vehicle_id)
            
            # Unpack result (handles both 2-tuple and 3-tuple returns)
            if len(ocr_result) == 3:
                final_plate_text, final_confidence, crop_path = ocr_result
            else:
                final_plate_text, final_confidence = ocr_result
                crop_path = None
            
            # 4. Check confidence threshold - reject if < 90%
            if final_confidence < 0.90:
                print(f"[ViolationRecorder] ❌ REJECTED: Plate confidence {final_confidence*100:.1f}% < 90%")
                print(f"[ViolationRecorder]   Plate text: {final_plate_text}")
                print(f"[ViolationRecorder]   Violation will NOT be reported")
                # Clean up files
                try:
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                except:
                    pass
                return
            
            # Categorize confidence
            if final_confidence >= 0.95:
                confidence_category = "HIGH (95%+)"
            elif final_confidence >= 0.90:
                confidence_category = "MEDIUM (90-94%)"
            else:
                confidence_category = "LOW (<90%)"  # Should not reach here due to check above
            
            print(f"[ViolationRecorder] ✅ Plate OCR successful:")
            print(f"[ViolationRecorder]   Text: {final_plate_text}")
            print(f"[ViolationRecorder]   Confidence: {final_confidence*100:.1f}% ({confidence_category})")
            
            # 5. Use cropped vehicle image as enhanced image (if available)
            if crop_path and os.path.exists(crop_path):
                print(f"[ViolationRecorder] Using cropped vehicle image as enhanced image")
                enhanced_plate_path = crop_path
            else:
                # Fallback: create enhanced image
                enhanced_plate_path = self._extract_and_enhance_plate(screenshot_path, vehicle_id, final_plate_text, bbox)
            
            # 4. Determine which video to use for upload
            # CRITICAL: Video evidence is mandatory - always get a video file
            video_for_upload = None
            
            if video_path == self.current_segment_path:
                print(f"[ViolationRecorder] Violation in CURRENT segment - checking duration")
                
                # Calculate how long the current segment has been recording
                current_time = time.time()
                current_duration = current_time - self.segment_start_time if self.segment_start_time else 0
                
                # If current segment is less than 5 seconds, use previous completed segment instead
                if current_duration < self.violation_clip_duration:
                    print(f"[ViolationRecorder] Current segment only {current_duration:.1f}s - using previous segment")
                    with self.segment_lock:
                        if self.video_segments:
                            video_for_upload = self.video_segments[-1]['path']
                            print(f"[ViolationRecorder] Using previous completed segment: {video_for_upload}")
                        else:
                            print(f"[ViolationRecorder] ⚠️ No previous segment available, waiting for current to reach 5s")
                else:
                    # Current segment has enough duration, wait for it to complete
                    print(f"[ViolationRecorder] Current segment has {current_duration:.1f}s - waiting for completion")
                    wait_time = 0
                    max_wait = 12
                    
                    while wait_time < max_wait:
                        time.sleep(1)
                        wait_time += 1
                        
                        # Check if segment has been completed (moved to video_segments)
                        with self.segment_lock:
                            for segment in reversed(self.video_segments):
                                if segment['path'] == video_path:
                                    print(f"[ViolationRecorder] ✅ Segment completed after {wait_time}s wait")
                                    video_for_upload = segment['path']
                                    break
                        
                        if video_for_upload:
                            break
                        
                        # Also check if a newer segment now contains our violation time
                        with self.segment_lock:
                            for segment in reversed(self.video_segments):
                                if segment['start_time'] <= violation_time <= segment['end_time']:
                                    print(f"[ViolationRecorder] ✅ Found violation in completed segment: {segment['path']}")
                                    video_for_upload = segment['path']
                                    break
                        
                        if video_for_upload:
                            break
                
                if not video_for_upload:
                    # Timeout - check if we should force close or use previous segment
                    current_time = time.time()
                    current_duration = current_time - self.segment_start_time if self.segment_start_time else 0
                    
                    if current_duration >= self.violation_clip_duration:
                        # Current segment has enough duration - FORCE close and use it
                        print(f"[ViolationRecorder] ⚠️ Timeout - forcing current segment to close ({current_duration:.1f}s)")
                        with self.segment_lock:
                            if self.current_writer is not None:
                                # Close the current writer to finalize the video file
                                try:
                                    self.current_writer.release()
                                except Exception as e:
                                    print(f"[ViolationRecorder] Warning releasing writer on forced close: {e}")

                                # Move tmp -> final if needed
                                try:
                                    if self.current_segment_tmp and os.path.exists(self.current_segment_tmp):
                                        os.replace(self.current_segment_tmp, self.current_segment_path)
                                except Exception as e:
                                    print(f"[ViolationRecorder] Warning moving tmp to final on forced close: {e}")

                                # Remux to ensure proper moov atom
                                try:
                                    if os.path.exists(self.current_segment_path):
                                        self._ffmpeg_remux(self.current_segment_path)
                                except Exception:
                                    pass

                                print(f"[ViolationRecorder] ✅ Forced segment close: {self.current_segment_path}")
                                
                                # Add this segment to completed segments
                                segment_info = {
                                    'path': self.current_segment_path,
                                    'start_time': self.segment_start_time,
                                    'end_time': time.time(),
                                    'frame_count': self.segment_frame_count,
                                    'duration': self.segment_frame_count / self.fps,
                                    'frames': self.frame_buffer.copy()
                                }
                                self.video_segments.append(segment_info)
                                
                                # Use this closed segment
                                video_for_upload = self.current_segment_path

                                # Clear current segment (will be recreated on next frame)
                                self.current_writer = None
                                self.current_segment_path = None
                                self.current_segment_tmp = None
                            elif self.video_segments:
                                # Current writer already closed, use most recent
                                video_for_upload = self.video_segments[-1]['path']
                                print(f"[ViolationRecorder] Using most recent completed segment: {video_for_upload}")
                    else:
                        # Current segment too short, use previous completed segment
                        print(f"[ViolationRecorder] ⚠️ Current segment only {current_duration:.1f}s - using previous segment")
                        with self.segment_lock:
                            if self.video_segments:
                                video_for_upload = self.video_segments[-1]['path']
                                print(f"[ViolationRecorder] Using previous completed segment: {video_for_upload}")
                            else:
                                print(f"[ViolationRecorder] ❌ No usable video available!")
            else:
                # Use completed segment directly
                video_for_upload = video_path
                print(f"[ViolationRecorder] Using completed segment: {video_for_upload}")
            
            if not video_for_upload:
                print(f"[ViolationRecorder] ❌ CRITICAL: No video available for violation!")
                print(f"[ViolationRecorder] Debug: video_path={video_path}, current_segment={self.current_segment_path}")
                print(f"[ViolationRecorder] Debug: completed_segments={len(self.video_segments) if self.video_segments else 0}")
                print(f"[ViolationRecorder] ❌ Cannot proceed - video evidence is mandatory")
                return
            
            # Verify video file exists and is readable
            if not os.path.exists(video_for_upload):
                print(f"[ViolationRecorder] ❌ Video file does not exist: {video_for_upload}")
                return
            
            # Trim video to 5-second clip around violation
            print(f"[ViolationRecorder] ✂️ Creating trimmed violation clip...")
            trimmed_video = self._trim_violation_video(video_for_upload, violation_time, vehicle_id)
            
            # 6. Upload trimmed video to Azure
            print(f"[ViolationRecorder] ⏱️ Starting video upload for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            video_url = self._upload_to_azure(trimmed_video, f"videos/{self.camera_id}/{vehicle_id}_{int(violation_time)}.mp4")
            
            if not video_url:
                print(f"[ViolationRecorder] ❌ Video upload failed - cannot proceed without video evidence")
                return
            
            # 7. Upload images to Azure
            print(f"[ViolationRecorder] ⏱️ Starting image uploads for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            screenshot_url = self._upload_to_azure(screenshot_path, f"images/{self.camera_id}/{vehicle_id}_{int(violation_time)}_original.jpg")
            enhanced_url = self._upload_to_azure(enhanced_plate_path, f"images/{self.camera_id}/{vehicle_id}_{int(violation_time)}_enhanced.jpg")
            print(f"[ViolationRecorder] ✅ All uploads completed for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            
            # Track files for this violation (for cleanup after successful upload)
            self.violation_files[vehicle_id] = []
            if screenshot_path and os.path.exists(screenshot_path):
                self.violation_files[vehicle_id].append(screenshot_path)
            if enhanced_plate_path and os.path.exists(enhanced_plate_path) and enhanced_plate_path != screenshot_path:
                self.violation_files[vehicle_id].append(enhanced_plate_path)
            # Add trimmed video for cleanup (only if different from segment)
            if trimmed_video and trimmed_video != video_for_upload and os.path.exists(trimmed_video):
                self.violation_files[vehicle_id].append(trimmed_video)
            # Note: video_for_upload (segment) not deleted immediately as it may be needed for other violations
            
            # 7. Build violation event with OCR results
            event = self._build_violation_event(
                vehicle_id, vehicle_class, speed_kmh, 
                final_plate_text, final_confidence, 
                violation_time, video_url, screenshot_url, enhanced_url
            )
            
            # 8. Send to backend API
            print(f"[ViolationRecorder] 📡 Sending violation API call for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            self._log_violation(vehicle_id, "API_SENDING", f"Plate: {final_plate_text} ({final_confidence*100:.1f}%)")
            api_success = self._send_violation_event(event, vehicle_id)
            print(f"[ViolationRecorder] {'✅' if api_success else '❌'} Violation API call completed for vehicle {vehicle_id} at {time.strftime('%H:%M:%S')}")
            
            if api_success:
                self._log_violation(vehicle_id, "API_SUCCESS", f"Plate: {final_plate_text}, Speed: {speed_kmh:.1f} km/h")
            else:
                self._log_violation(vehicle_id, "API_FAILED", f"Plate: {final_plate_text}")
            
            print(f"[ViolationRecorder] ✅ Violation processed successfully for vehicle {vehicle_id}")
            print(f"[ViolationRecorder]   Plate: {final_plate_text} ({final_confidence*100:.1f}%)")
            
        except Exception as e:
            print(f"[ViolationRecorder] ❌ Error processing violation: {e}")
            self._log_violation(vehicle_id, "ERROR", str(e)[:100])
            import traceback
            traceback.print_exc()
    
    def _get_violation_video(self, violation_time):
        """
        Find video segment containing the violation timestamp.
        For current segment, we'll just use the path directly.
        """
        with self.segment_lock:
            # First, check completed segments
            for segment in reversed(self.video_segments):
                if segment['start_time'] <= violation_time <= segment['end_time']:
                    print(f"[ViolationRecorder] Found violation in completed segment: {segment['path']}")
                    return segment['path']
            
            # Check if violation is in the CURRENT recording segment
            if self.current_segment_path and self.segment_start_time:
                current_time = time.time()
                if self.segment_start_time <= violation_time <= current_time:
                    print(f"[ViolationRecorder] Violation is in CURRENT recording segment: {self.current_segment_path}")
                    # Just return the current segment path - it will be used as-is
                    return self.current_segment_path
        
        # If exact segment not found, use most recent completed segment
        if self.video_segments:
            recent_segment = self.video_segments[-1]['path']
            print(f"[ViolationRecorder] Using most recent completed segment: {recent_segment}")
            return recent_segment
        
        # Last resort: use current segment if it exists
        if self.current_segment_path and os.path.exists(self.current_segment_path):
            print(f"[ViolationRecorder] Using current segment (last resort): {self.current_segment_path}")
            return self.current_segment_path
        
        print(f"[ViolationRecorder] ❌ No video segment found for violation time {violation_time}")
        return None
    
    def _trim_violation_video(self, source_video_path, violation_time, vehicle_id):
        """
        Trim video to extract 5-second clip around violation timestamp.
        
        Args:
            source_video_path: Path to the source video file
            violation_time: Unix timestamp of the violation
            vehicle_id: Vehicle tracking ID
            
        Returns:
            Path to trimmed video file, or source_video_path if trimming fails
        """
        try:
            # Read the source video
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                print(f"[ViolationRecorder] ⚠️ Cannot open video for trimming: {source_video_path}")
                return source_video_path
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate video duration
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[ViolationRecorder] Trimming video:")
            print(f"[ViolationRecorder]   Source: {source_video_path}")
            print(f"[ViolationRecorder]   Duration: {duration:.2f}s, FPS: {fps}, Frames: {total_frames}")
            
            # Calculate violation offset within the segment
            with self.segment_lock:
                # Find which segment this video belongs to
                segment_start_time = None
                for seg in self.video_segments:
                    if seg['path'] == source_video_path:
                        segment_start_time = seg['start_time']
                        break
                
                # If not found in completed segments, use current segment start
                if segment_start_time is None and source_video_path == self.current_segment_path:
                    segment_start_time = self.segment_start_time
            
            if segment_start_time is None:
                print(f"[ViolationRecorder] ⚠️ Cannot determine segment start time")
                cap.release()
                return source_video_path
            
            # Calculate violation offset from segment start
            violation_offset = violation_time - segment_start_time
            
            print(f"[ViolationRecorder]   Violation offset: {violation_offset:.2f}s into segment")
            
            # Define trim range: 2.5s before + 2.5s after = 5s total
            trim_duration = self.violation_clip_duration  # 5 seconds
            half_duration = trim_duration / 2  # 2.5 seconds
            
            trim_start = max(0, violation_offset - half_duration)
            trim_end = min(duration, violation_offset + half_duration)
            
            # Adjust if we're at the edges
            if trim_start == 0:
                trim_end = min(duration, trim_duration)
            if trim_end == duration:
                trim_start = max(0, duration - trim_duration)
            
            actual_duration = trim_end - trim_start
            
            print(f"[ViolationRecorder]   Trim range: {trim_start:.2f}s to {trim_end:.2f}s ({actual_duration:.2f}s)")
            
            # Calculate frame range
            start_frame = int(trim_start * fps)
            end_frame = int(trim_end * fps)
            frames_to_write = end_frame - start_frame
            
            if frames_to_write <= 0:
                print(f"[ViolationRecorder] ⚠️ Invalid frame range: {start_frame} to {end_frame}")
                cap.release()
                return source_video_path
            
            # Create output file path
            trimmed_path = os.path.join(self.proof_dir, f"trimmed_{vehicle_id}_{int(violation_time)}.mp4")

            # Use ffmpeg to trim (copy codec) and add faststart
            try:
                # Best-effort: remux source first to ensure moov atom (helps OpenCV/ffmpeg reads)
                try:
                    self._ffmpeg_remux(source_video_path)
                except Exception:
                    pass

                # Primary approach: re-encode trimmed clip to ensure a valid MP4 (more robust)
                reencode_cmd = [
                    'ffmpeg', '-y', '-i', str(source_video_path),
                    '-ss', str(trim_start), '-t', str(actual_duration),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(trimmed_path)
                ]

                try:
                    subprocess.run(reencode_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Fallback: try a copy-based trim (faster) if re-encode fails
                    copy_cmd = [
                        'ffmpeg', '-y', '-i', str(source_video_path),
                        '-ss', str(trim_start), '-t', str(actual_duration),
                        '-c', 'copy', '-movflags', '+faststart', str(trimmed_path)
                    ]
                    subprocess.run(copy_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Verify trimmed output
                if os.path.exists(trimmed_path):
                    file_size = os.path.getsize(trimmed_path)
                    file_size_mb = file_size / (1024 * 1024)

                    # If file is suspiciously small, consider trimming failed
                    if file_size < 1024:
                        print(f"[ViolationRecorder] ⚠️ Trimmed video {trimmed_path} is very small ({file_size} bytes), likely invalid")
                        try:
                            os.remove(trimmed_path)
                        except Exception:
                            pass
                        return source_video_path

                    print(f"[ViolationRecorder] ✅ Trimmed video created: {trimmed_path}")
                    print(f"[ViolationRecorder]   Duration: {actual_duration:.2f}s, Frames: {frames_to_write}, Size: {file_size_mb:.2f} MB")
                    # Add to violation files for cleanup tracking
                    if vehicle_id in self.violation_files:
                        self.violation_files[vehicle_id].append(trimmed_path)
                    return trimmed_path
                else:
                    print(f"[ViolationRecorder] ❌ Trimmed video file was not created by ffmpeg")
                    return source_video_path

            except Exception as e:
                print(f"[ViolationRecorder] ❌ Error trimming video with ffmpeg: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return source
                return source_video_path
                
        except Exception as e:
            print(f"[ViolationRecorder] ❌ Error trimming video: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original video if trimming fails
            return source_video_path
    
    def _capture_violation_screenshot(self, violation_time, vehicle_id, vehicle_class="vehicle", speed_kmh=0):
        """Capture screenshot at violation moment with violation info overlay."""
        # Find frame closest to violation time
        target_frame = None
        min_time_diff = float('inf')
        
        with self.segment_lock:
            # Check completed segments
            for segment in self.video_segments:
                for frame_data in segment['frames']:
                    time_diff = abs(frame_data['timestamp'] - violation_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        target_frame = frame_data['frame'].copy()  # Use copy to add overlay
            
            # Also check CURRENT frame buffer (might not be in completed segments yet)
            for frame_data in self.frame_buffer:
                time_diff = abs(frame_data['timestamp'] - violation_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    target_frame = frame_data['frame'].copy()
                    print(f"[ViolationRecorder] Found frame in CURRENT buffer (time_diff: {time_diff:.3f}s)")
        
        if target_frame is None:
            print(f"[ViolationRecorder] ❌ No frame found near violation time")
            return None
            print(f"[ViolationRecorder] No frame found near violation time")
            return None
        
        # Add violation info overlay
        height, width = target_frame.shape[:2]
        
        # Add solid red banner at top (100% opacity, no transparency)
        cv2.rectangle(target_frame, (0, 0), (width, 120), (0, 0, 180), -1)
        
        # Add violation text (use FONT_HERSHEY_SIMPLEX with bold thickness)
        violation_date = datetime.fromtimestamp(violation_time, tz=timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(target_frame, "SPEED VIOLATION", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(target_frame, f"Speed: {int(speed_kmh)} km/h  |  Limit: {self.speed_limit} km/h", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(target_frame, f"Vehicle: {vehicle_class.upper()}  |  Time: {violation_date}", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Save screenshot with overlay
        screenshot_path = self.proof_dir / f"violation_{vehicle_id}_{int(violation_time)}_original.jpg"
        cv2.imwrite(str(screenshot_path), target_frame)
        print(f"[ViolationRecorder] Screenshot saved with overlay: {screenshot_path}")
        
        return screenshot_path
    
    def _capture_screenshot_from_frame(self, frame, bbox, vehicle_id, vehicle_class, speed_kmh, violation_time):
        """
        Capture screenshot from provided frame with violation overlay.
        
        Args:
            frame: OpenCV frame (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
            vehicle_id: Vehicle ID
            vehicle_class: Vehicle class
            speed_kmh: Speed in km/h
            violation_time: Timestamp
        """
        if frame is None:
            print(f"[ViolationRecorder] No frame provided for screenshot")
            return None
        
        try:
            screenshot = frame.copy()
            
            # Add violation info overlay (solid red banner at top - 100% opacity)
            height, width = screenshot.shape[:2]
            
            # Solid red banner (no transparency)
            cv2.rectangle(screenshot, (0, 0), (width, 80), (0, 0, 200), -1)
            
            # Violation text
            violation_text = f"SPEED VIOLATION - {speed_kmh:.1f} km/h (Limit: {self.speed_limit} km/h)"
            cv2.putText(screenshot, violation_text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Date/time and vehicle class (Indian Standard Time)
            time_str = datetime.fromtimestamp(violation_time, tz=timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S')
            info_text = f"{time_str} | Vehicle: {vehicle_class.upper()} | ID: {vehicle_id}"
            cv2.putText(screenshot, info_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Highlight vehicle with green box if bbox available
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(screenshot, f"#{vehicle_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save screenshot
            screenshot_path = self.proof_dir / f"violation_{vehicle_id}_{int(violation_time)}.jpg"
            cv2.imwrite(str(screenshot_path), screenshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"[ViolationRecorder] Screenshot saved: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            print(f"[ViolationRecorder] Error capturing screenshot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_ocr_on_vehicle(self, screenshot_path, bbox, vehicle_id):
        """
        Run OCR using Gemini Vision API on cropped vehicle region.
        
        Args:
            screenshot_path: Path to screenshot image
            bbox: Bounding box of vehicle (x1, y1, x2, y2)
            vehicle_id: Vehicle ID for logging
        
        Returns:
            tuple: (plate_text, confidence) - Returns ("UNKNOWN", 0.0) if OCR fails
        """
        if not GEMINI_AVAILABLE or self.gemini_ocr is None:
            print(f"[ViolationRecorder] Gemini OCR not available")
            return ("UNKNOWN", 0.0)
        
        if bbox is None:
            print(f"[ViolationRecorder] No bounding box provided for OCR")
            return ("UNKNOWN", 0.0)
        
        try:
            # Read original image
            img = cv2.imread(str(screenshot_path))
            if img is None:
                print(f"[ViolationRecorder] Failed to read screenshot")
                return ("UNKNOWN", 0.0)
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Crop to FULL vehicle bounding box with padding
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Add padding to ensure full car is visible
            h_pad = 30  # Horizontal padding
            v_pad = 30  # Vertical padding
            
            x1_crop = max(0, x1 - h_pad)
            y1_crop = max(0, y1 - v_pad)
            x2_crop = min(img_width, x2 + h_pad)
            y2_crop = min(img_height, y2 + v_pad)
            
            # Crop to full vehicle + padding
            vehicle_crop = img[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
            if vehicle_crop.size == 0:
                print(f"[ViolationRecorder] Invalid crop region")
                return ("UNKNOWN", 0.0)
            
            print(f"[ViolationRecorder] Cropped FULL vehicle: {vehicle_crop.shape}")
            print(f"[ViolationRecorder] Crop coords: x={x1_crop}:{x2_crop}, y={y1_crop}:{y2_crop}")
            
            # Save cropped image (this will be used as enhanced image for upload)
            crop_path = self.proof_dir / f"vehicle_crop_{vehicle_id}_{int(time.time())}.jpg"
            cv2.imwrite(str(crop_path), vehicle_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"[ViolationRecorder] 🔍 CROPPED IMAGE SAVED → {crop_path.absolute()}")
            print(f"[ViolationRecorder] 📁 This image will be sent to Gemini OCR and uploaded to Azure!")
            
            # Prepare Gemini prompt for license plate detection
            prompt = """Analyze this vehicle image and extract the license plate number.

IMPORTANT REQUIREMENTS:
1. License plates in this footage have exactly 7 alphanumeric characters
2. Format: 7 characters total (mix of letters and numbers)
3. Examples: ABC1234, XYZ9876, 1AB2345, DEF0987
4. Return ONLY the 7-character plate number (no spaces, no dashes)
5. If multiple plates visible, return the MOST VISIBLE/CLEAR one
6. If no valid 7-character license plate found, return exactly: "UNKNOWN"

Respond in JSON format:
{
  "plate_text": "ABC1234",
  "confidence": 0.95,
  "reasoning": "Clear front plate visible with all 7 characters readable"
}

Confidence scale:
- 0.95-1.0: Plate is very clear, all 7 characters are readable and certain
- 0.90-0.94: Plate is readable with minor blur/angle, confident on 6-7 characters
- 0.80-0.89: Plate partially obscured or angled, can read 5-6 characters clearly
- <0.80: Plate not clearly visible, less than 5 characters readable

Be strict with confidence scoring. Only give 90%+ if you can clearly read at least 6 out of 7 characters."""

            # Call Gemini Vision API
            print(f"[ViolationRecorder] Calling Gemini Vision API for OCR...")
            
            # Create PIL Image from numpy array for Gemini
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
            
            response = self.gemini_ocr.generate_content([prompt, pil_image])
            
            if not response or not response.text:
                print(f"[ViolationRecorder] No response from Gemini")
                return ("UNKNOWN", 0.0)
            
            print(f"[ViolationRecorder] Gemini response: {response.text}")
            
            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                response_text = response.text.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                
                plate_text = result.get('plate_text', 'UNKNOWN').strip().upper()
                confidence = float(result.get('confidence', 0.0))
                reasoning = result.get('reasoning', '')
                
                print(f"[ViolationRecorder] Gemini OCR result:")
                print(f"[ViolationRecorder]   Plate: {plate_text}")
                print(f"[ViolationRecorder]   Confidence: {confidence*100:.1f}%")
                print(f"[ViolationRecorder]   Reasoning: {reasoning}")
                
                # Validate format if not UNKNOWN
                if plate_text != "UNKNOWN":
                    # Clean up text
                    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text)
                    
                    # Validate Indian plate format
                    if self._is_valid_plate_format(cleaned):
                        formatted_plate = self._format_plate(cleaned)
                        print(f"[ViolationRecorder] ✅ Valid plate format: {formatted_plate}")
                        return (formatted_plate, confidence, str(crop_path))  # Return crop path too
                    else:
                        print(f"[ViolationRecorder] ⚠️ Invalid plate format: {cleaned}")
                        return ("UNKNOWN", 0.0, str(crop_path))
                else:
                    return ("UNKNOWN", 0.0, str(crop_path))
                    
            except json.JSONDecodeError as e:
                print(f"[ViolationRecorder] Failed to parse Gemini response as JSON: {e}")
                print(f"[ViolationRecorder] Raw response: {response.text}")
                return ("UNKNOWN", 0.0, str(crop_path))
                
        except Exception as e:
            print(f"[ViolationRecorder] Gemini OCR error: {e}")
            import traceback
            traceback.print_exc()
            return ("UNKNOWN", 0.0, None)
    
    def _is_valid_plate_format(self, text):
        """
        Validate 7-character license plate format.
        Format: Exactly 7 alphanumeric characters (letters and/or numbers)
        Examples: ABC1234, XYZ9876, 1AB2345, DEF0987
        """
        # Must be exactly 7 characters
        if len(text) != 7:
            return False
        
        # Must be alphanumeric only
        if not text.isalnum():
            return False
        
        # Must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number
    
    def _format_plate(self, text):
        """
        Format plate text for display (7 characters, no spaces).
        Example: ABC1234 -> ABC1234
        """
        # Return as-is (7 characters, no spaces needed)
        return text.upper()
    
    def _extract_and_enhance_plate(self, screenshot_path, vehicle_id, plate_text, bbox=None):
        """
        Create enhanced image focused on vehicle region (no heavy processing).
        """
        print(f"[ViolationRecorder] Creating vehicle-focused image...")
        
        # Read original image
        img = cv2.imread(str(screenshot_path))
        if img is None:
            return None
        
        # If bbox provided, crop to vehicle region (lower half where plate is)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            img_height, img_width = img.shape[:2]
            
            # Focus on lower half of vehicle
            bbox_height = y2 - y1
            y1_plate = y1 + int(bbox_height * 0.5)
            
            # Add padding
            h_pad = 40
            v_pad = 20
            
            x1 = max(0, x1 - h_pad)
            y1 = max(0, y1_plate - v_pad)
            x2 = min(img_width, x2 + h_pad)
            y2 = min(img_height, y2 + v_pad)
            
            img = img[y1:y2, x1:x2].copy()
        
        # Save as high-quality JPEG (no enhancement, just crop)
        enhanced_path = self.proof_dir / f"violation_{vehicle_id}_{int(time.time())}_enhanced.jpg"
        cv2.imwrite(str(enhanced_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"[ViolationRecorder] Vehicle-focused image saved: {enhanced_path}")
        
        return enhanced_path
    
    def _upload_to_azure(self, file_path, blob_name):
        """
        Upload file to Azure Blob Storage.
        
        Args:
            file_path: Local file path
            blob_name: Blob name in container (e.g., "videos/CAM-001/vehicle123.mp4")
        
        Returns:
            str: Public URL of uploaded blob, or local path if Azure unavailable
        """
        if not AZURE_AVAILABLE or not self.azure_connection_string:
            print(f"[ViolationRecorder] Azure not configured, using local path: {file_path}")
            return f"file://{file_path}"
        
        if not file_path or not os.path.exists(file_path):
            print(f"[ViolationRecorder] ⚠️ File does not exist: {file_path}")
            return f"file://{file_path}"
        
        try:
            # Create blob client
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            blob_client = blob_service_client.get_blob_client(
                container=self.azure_container_name,
                blob=blob_name
            )
            
            # Upload file with retry on file access issues
            print(f"[ViolationRecorder] Uploading to Azure: {blob_name}")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    break  # Success
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"[ViolationRecorder] File access error (attempt {attempt+1}/{max_retries}), retrying...")
                        time.sleep(0.5)  # Wait and retry
                    else:
                        raise  # Give up after max retries
            
            # Get URL
            blob_url = blob_client.url
            print(f"[ViolationRecorder] ✅ Uploaded to Azure: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            print(f"[ViolationRecorder] ❌ Azure upload failed: {e}")
            return f"file://{file_path}"
    
    def _build_violation_event(self, vehicle_id, vehicle_class, speed_kmh, 
                               plate_text, plate_confidence, violation_time,
                               video_url, screenshot_url, enhanced_url):
        """Build violation event JSON matching new API format."""
        # Format timestamp in Indian Standard Time (IST = UTC+5:30)
        ist_timezone = timezone(timedelta(hours=5, minutes=30))
        captured_at = datetime.fromtimestamp(violation_time, tz=ist_timezone).strftime('%Y-%m-%dT%H:%M:%S')
        
        event = {
            "eventId": f"EVT-{uuid.uuid4().hex[:8].upper()}",
            "cameraId": self.camera_id,
            "capturedAt": captured_at,
            "evidence": {
                "imageOriginalUrl": screenshot_url,
                "imageEnhancedUrl": enhanced_url,
                "videoClipUrl": video_url  # Mandatory - always included
            },
            "violation": {
                "type": "OVERSPEED",
                "measured": round(speed_kmh, 1),
                "limit": float(self.speed_limit)
            },
            "vehicle": {
                "vehicleClass": vehicle_class.upper(),
                "plate": {
                    "text": plate_text or "UNKNOWN",
                    "confidence": round(plate_confidence, 2)
                }
            }
        }
        
        # HMAC signature removed - not needed by API
        
        return event
    
    def _send_violation_event(self, event, vehicle_id=None):
        """
        Save violation event to local SQLite database.
        
        Args:
            event: Violation event JSON
            vehicle_id: Vehicle ID for cleanup tracking (optional)
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            print(f"[ViolationRecorder] Saving violation to local database")
            
            # Remove hash field - not needed
            event_copy = event.copy()
            event_copy.pop('hash', None)
            
            print(f"[ViolationRecorder] Event: {json.dumps(event_copy, indent=2)}")
            
            # Save to local SQLite database
            success = db.add_violation(event_copy)
            
            if success:
                print(f"[ViolationRecorder] ✅ Violation saved to database successfully")
                
                # Note: We keep local proof files for serving via /api/proof/ endpoint
                # Don't delete files since we need them for the violations page
                
                return True
            else:
                print(f"[ViolationRecorder] ❌ Failed to save violation to database")
                return False
                
        except Exception as e:
            print(f"[ViolationRecorder] ❌ Failed to save violation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup_old_files_loop(self):
        """Background thread that periodically cleans up old files in proof directory."""
        print(f"[ViolationRecorder] 🧹 Cleanup thread started - runs every 2 minutes")
        
        iteration = 0
        while self.cleanup_running:
            try:
                # Run cleanup every 2 minutes (120 seconds)
                time.sleep(120)
                iteration += 1
                print(f"[ViolationRecorder] 🔄 Running scheduled cleanup #{iteration} (every 2 min)")
                self._cleanup_old_files()
            except Exception as e:
                print(f"[ViolationRecorder] Cleanup thread error: {e}")
    
    def _cleanup_old_files(self):
        """Delete files older than 10 minutes from proof directory."""
        try:
            current_time = time.time()
            cleanup_threshold = 600  # 10 minutes in seconds
            deleted_count = 0
            total_files = 0
            
            # Iterate through all files in proof directory
            for file_path in self.proof_dir.glob('*'):
                if file_path.is_file():
                    total_files += 1
                    # Get file modification time
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > cleanup_threshold:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            print(f"[ViolationRecorder]   🗑️ Deleted: {file_path.name} (age: {file_age/60:.1f}min)")
                        except Exception as e:
                            print(f"[ViolationRecorder] Failed to delete {file_path.name}: {e}")
            
            # Always log cleanup result (even if nothing deleted)
            print(f"[ViolationRecorder] 🧹 Cleanup complete: {deleted_count}/{total_files} files deleted (threshold: 10min)")
            self._log_violation(0, "CLEANUP_RUN", f"Deleted {deleted_count}/{total_files} old files")
                
        except Exception as e:
            print(f"[ViolationRecorder] Cleanup error: {e}")
    
    def _delete_violation_files(self, vehicle_id):
        """Delete all files associated with a successfully uploaded violation."""
        if vehicle_id not in self.violation_files:
            return
        
        deleted_count = 0
        for file_path in self.violation_files[vehicle_id]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"[ViolationRecorder] 🗑️ Deleted: {Path(file_path).name}")
            except Exception as e:
                print(f"[ViolationRecorder] Failed to delete {file_path}: {e}")
        
        # Remove from tracking
        del self.violation_files[vehicle_id]
        
        if deleted_count > 0:
            print(f"[ViolationRecorder] ✅ Cleaned up {deleted_count} files for vehicle {vehicle_id}")
    
    def cleanup(self):
        """Clean up resources."""
        print(f"[ViolationRecorder] Starting cleanup...")
        
        # Stop cleanup thread
        self.cleanup_running = False
        
        # Stop violation workers gracefully
        self.workers_running = False
        print(f"[ViolationRecorder] Waiting for {self.violation_queue.qsize()} pending violations to finish...")
        
        # Send poison pills to workers
        for _ in range(self.processing_workers):
            try:
                self.violation_queue.put(None, timeout=1)
            except:
                pass
        
        # Wait for workers to finish (max 30 seconds)
        for i, worker in enumerate(self.worker_threads):
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"[ViolationRecorder] Worker {i} did not finish in time")
        
        # Log final stats
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
        except:
            pass
        
        if self.current_writer is not None:
            self.current_writer.release()
        
        print(f"[ViolationRecorder] Cleanup complete")
        print(f"[ViolationRecorder] Cleaned up")


# Helper function to integrate with existing speed detection
def create_violation_recorder(camera_id, speed_limit, line_a_y=None, line_b_y=None):
    """
    Factory function to create violation recorder instance.
    
    Usage in calib_server.py:
        recorder = create_violation_recorder("CAM-CHD-042", 75, line_a_y=300, line_b_y=500)
        
        # In frame processing loop:
        recorder.add_frame(frame)
        
        # When speed is calculated (pass bbox and frame for middle line trigger):
        if recorder.check_violation(vehicle_id, "car", 85.3, bbox=(x1,y1,x2,y2), frame=current_frame):
            print("Violation recorded!")
    """
    return ViolationRecorder(camera_id, speed_limit, line_a_y, line_b_y)

