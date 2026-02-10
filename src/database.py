"""
NTCS Local SQLite Database Module
Replaces external backend API (nextgen-fv1h.onrender.com) with a local SQLite database.
Stores cameras, calibrations, and violations locally.
"""

import os
import json
import sqlite3
import threading
from datetime import datetime

# Database file path (next to this script, inside src/)
DB_PATH = os.path.join(os.path.dirname(__file__), 'ntcs.db')

# Thread-local storage for connections
_local = threading.local()


def _get_conn():
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Initialize database tables. Safe to call multiple times."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT UNIQUE NOT NULL,
            speed_limit    INTEGER NOT NULL,
            created_at     TEXT DEFAULT (datetime('now')),
            updated_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS cameras (
            camera_id       TEXT PRIMARY KEY,
            camera_link     TEXT NOT NULL,
            location        TEXT NOT NULL,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS calibrations (
            camera_id       TEXT PRIMARY KEY,
            camera_link     TEXT,
            location        TEXT,
            current_status  TEXT DEFAULT 'INACTIVE',
            speed_limit     INTEGER DEFAULT 75,
            line_ay         REAL,
            line_by         REAL,
            polygon_points  TEXT,   -- JSON array
            distance        REAL,
            width           REAL,
            method          TEXT,
            confidence      REAL,
            reason          TEXT,
            updated_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS violations (
            event_id            TEXT PRIMARY KEY,
            camera_id           TEXT,
            captured_at         TEXT,
            image_original_url  TEXT,
            image_enhanced_url  TEXT,
            video_clip_url      TEXT,
            violation_type      TEXT DEFAULT 'OVERSPEED',
            measured_speed      REAL,
            speed_limit         REAL,
            vehicle_class       TEXT,
            plate_text          TEXT,
            plate_confidence    REAL,
            created_at          TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_violations_camera ON violations(camera_id);
        CREATE INDEX IF NOT EXISTS idx_violations_captured ON violations(captured_at);
        CREATE INDEX IF NOT EXISTS idx_violations_plate ON violations(plate_text);
    """)
    conn.commit()
    print(f"[Database] Initialized SQLite database at: {DB_PATH}")


# ==========================
#  Locations CRUD
# ==========================

def get_locations():
    """Return list of all locations with speed limits."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT location_id, name, speed_limit, created_at, updated_at FROM locations ORDER BY name ASC"
    ).fetchall()
    return [
        {
            "locationId": r["location_id"],
            "name": r["name"],
            "speedLimit": r["speed_limit"],
            "createdAt": r["created_at"],
            "updatedAt": r["updated_at"],
        }
        for r in rows
    ]


def add_location(name, speed_limit):
    """Insert a new location. Returns True on success."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO locations (name, speed_limit, created_at, updated_at)
               VALUES (?, ?, datetime('now'), datetime('now'))""",
            (name, speed_limit),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database] Error adding location: {e}")
        return False


def get_location_by_id(location_id):
    """Get a location row by ID."""
    conn = _get_conn()
    return conn.execute(
        "SELECT location_id, name, speed_limit FROM locations WHERE location_id = ?",
        (location_id,),
    ).fetchone()


def update_location(location_id, name, speed_limit):
    """Update an existing location. Returns True if updated."""
    conn = _get_conn()
    try:
        row = get_location_by_id(location_id)
        if not row:
            return False
        old_name = row["name"]
        cur = conn.execute(
            """UPDATE locations
               SET name = ?, speed_limit = ?, updated_at = datetime('now')
               WHERE location_id = ?""",
            (name, speed_limit, location_id),
        )
        # Keep camera and calibration references in sync if name changed
        if old_name != name:
            conn.execute(
                "UPDATE cameras SET location = ? WHERE location = ?",
                (name, old_name),
            )
            conn.execute(
                "UPDATE calibrations SET location = ? WHERE location = ?",
                (name, old_name),
            )
        # Always update speed limits for cameras tied to this location
        conn.execute(
            "UPDATE calibrations SET speed_limit = ?, updated_at = datetime('now') WHERE location = ?",
            (speed_limit, name),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        print(f"[Database] Error updating location: {e}")
        return False


def delete_location(location_id):
    """Delete a location. Returns True if deleted."""
    conn = _get_conn()
    row = get_location_by_id(location_id)
    if not row:
        return False, "not_found"
    name = row["name"]
    used = conn.execute(
        "SELECT COUNT(*) as cnt FROM cameras WHERE location = ?",
        (name,),
    ).fetchone()["cnt"]
    if used and used > 0:
        return False, "in_use"
    cur = conn.execute(
        "DELETE FROM locations WHERE location_id = ?",
        (location_id,),
    )
    conn.commit()
    return cur.rowcount > 0, "deleted"


def get_location_by_name(name):
    """Get a location row by name."""
    conn = _get_conn()
    return conn.execute(
        "SELECT location_id, name, speed_limit FROM locations WHERE name = ?",
        (name,),
    ).fetchone()


def get_location_speed_limit(name):
    """Get speed limit for a location name. Returns None if not found."""
    row = get_location_by_name(name)
    if row:
        return row["speed_limit"]
    return None


def ensure_calibration_row(camera_id, camera_link, location, speed_limit):
    """Ensure a calibration row exists for a camera with location/speed limit."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO calibrations
                   (camera_id, camera_link, location, current_status, speed_limit,
                    line_ay, line_by, distance, width, method, confidence, reason, updated_at)
               VALUES (?, ?, ?, 'INACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(camera_id) DO UPDATE SET
                   camera_link = excluded.camera_link,
                   location = excluded.location,
                   speed_limit = excluded.speed_limit,
                   updated_at = datetime('now')""",
            (
                camera_id,
                camera_link,
                location,
                speed_limit,
                300,
                500,
                10.0,
                10.0,
                'unknown',
                0.0,
                'Initialized defaults',
            ),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database] Error ensuring calibration row: {e}")
        return False


# ==========================
#  Camera CRUD
# ==========================

def get_camera(camera_id):
    """Return a single camera by ID or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT camera_id, camera_link, location, created_at FROM cameras WHERE camera_id = ?",
        (camera_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "cameraId": row["camera_id"],
        "cameraLink": row["camera_link"],
        "location": row["location"],
        "createdAt": row["created_at"],
    }

def get_cameras():
    """Return list of all cameras as dicts."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT camera_id, camera_link, location, created_at FROM cameras ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "cameraId": r["camera_id"],
            "cameraLink": r["camera_link"],
            "location": r["location"],
            "createdAt": r["created_at"],
        }
        for r in rows
    ]


def add_camera(camera_id, camera_link, location):
    """Insert or update a camera. Returns True on success."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO cameras (camera_id, camera_link, location)
               VALUES (?, ?, ?)
               ON CONFLICT(camera_id) DO UPDATE SET
                   camera_link = excluded.camera_link,
                   location = excluded.location""",
            (camera_id, camera_link, location),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Database] Error adding camera: {e}")
        return False


def delete_camera(camera_id):
    """Delete a camera (cascades to calibration). Returns True if deleted."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
    conn.commit()
    return cur.rowcount > 0


# ==========================
#  Calibration CRUD
# ==========================

def get_calibration(camera_id):
    """
    Get calibration for a camera.
    Returns dict matching the old API shape, or None.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM calibrations WHERE camera_id = ?", (camera_id,)
    ).fetchone()
    if not row:
        return None

    polygon_points = None
    if row["polygon_points"]:
        try:
            polygon_points = json.loads(row["polygon_points"])
        except Exception:
            polygon_points = None

    return {
        "cameraId": row["camera_id"],
        "cameraLink": row["camera_link"],
        "location": row["location"],
        "currentStatus": row["current_status"],
        "speedLimit": row["speed_limit"],
        "stats": {
            "line_ay": row["line_ay"],
            "line_by": row["line_by"],
            "polygon_points": polygon_points,
            "distance": row["distance"],
            "width": row["width"],
            "method": row["method"],
            "confidence": row["confidence"],
            "reason": row["reason"],
        },
    }


def save_calibration(camera_id, camera_link, location, stats, status="ACTIVE", speed_limit=75):
    """Save or update calibration data."""
    conn = _get_conn()
    polygon_json = None
    if stats.get("polygon_points"):
        polygon_json = json.dumps(stats["polygon_points"])

    try:
        conn.execute(
            """INSERT INTO calibrations
                   (camera_id, camera_link, location, current_status, speed_limit,
                    line_ay, line_by, polygon_points, distance, width,
                    method, confidence, reason, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(camera_id) DO UPDATE SET
                   camera_link = excluded.camera_link,
                   location = excluded.location,
                   current_status = excluded.current_status,
                   speed_limit = excluded.speed_limit,
                   line_ay = excluded.line_ay,
                   line_by = excluded.line_by,
                   polygon_points = excluded.polygon_points,
                   distance = excluded.distance,
                   width = excluded.width,
                   method = excluded.method,
                   confidence = excluded.confidence,
                   reason = excluded.reason,
                   updated_at = datetime('now')""",
            (
                camera_id,
                camera_link,
                location,
                status,
                speed_limit,
                stats.get("line_ay"),
                stats.get("line_by"),
                polygon_json,
                stats.get("distance"),
                stats.get("width"),
                stats.get("method"),
                stats.get("confidence"),
                stats.get("reason"),
            ),
        )
        conn.commit()
        print(f"[Database] Calibration saved for {camera_id}")
        return True
    except Exception as e:
        print(f"[Database] Error saving calibration: {e}")
        return False


def get_speed_limit(camera_id):
    """Get speed limit for camera. Returns 75 as default."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT speed_limit FROM calibrations WHERE camera_id = ?", (camera_id,)
    ).fetchone()
    if row and row["speed_limit"]:
        return row["speed_limit"]
    return 75


def update_speed_limit(camera_id, speed_limit):
    """Update speed limit for a camera."""
    conn = _get_conn()
    conn.execute(
        "UPDATE calibrations SET speed_limit = ?, updated_at = datetime('now') WHERE camera_id = ?",
        (speed_limit, camera_id),
    )
    conn.commit()


def update_camera_status_db(camera_id, status):
    """Update camera status (ACTIVE/INACTIVE)."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE calibrations SET current_status = ?, updated_at = datetime('now') WHERE camera_id = ?",
        (status, camera_id),
    )
    conn.commit()
    return cur.rowcount > 0


def check_camera_active(camera_id):
    """Check if camera is ACTIVE."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT current_status FROM calibrations WHERE camera_id = ?", (camera_id,)
    ).fetchone()
    return row is not None and row["current_status"] == "ACTIVE"


# ==========================
#  Violation CRUD
# ==========================

def add_violation(event):
    """
    Insert a violation event. Expects the same JSON shape as the old API.
    Returns True on success.
    """
    conn = _get_conn()
    try:
        evidence = event.get("evidence", {})
        violation = event.get("violation", {})
        vehicle = event.get("vehicle", {})
        plate = vehicle.get("plate", {})

        conn.execute(
            """INSERT INTO violations
                   (event_id, camera_id, captured_at,
                    image_original_url, image_enhanced_url, video_clip_url,
                    violation_type, measured_speed, speed_limit,
                    vehicle_class, plate_text, plate_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.get("eventId"),
                event.get("cameraId"),
                event.get("capturedAt"),
                evidence.get("imageOriginalUrl"),
                evidence.get("imageEnhancedUrl"),
                evidence.get("videoClipUrl"),
                violation.get("type", "OVERSPEED"),
                violation.get("measured"),
                violation.get("limit"),
                vehicle.get("vehicleClass"),
                plate.get("text"),
                plate.get("confidence"),
            ),
        )
        conn.commit()
        print(f"[Database] Violation saved: {event.get('eventId')}")
        return True
    except sqlite3.IntegrityError:
        print(f"[Database] Violation {event.get('eventId')} already exists (duplicate)")
        return True  # Not an error, just a duplicate
    except Exception as e:
        print(f"[Database] Error saving violation: {e}")
        return False


def get_violations(camera_id=None, page=1, per_page=20, plate_text=None):
    """
    Get violations with pagination and optional filters.
    Returns { violations: [...], total: int, page: int, per_page: int, pages: int }
    """
    conn = _get_conn()
    conditions = []
    params = []

    if camera_id:
        conditions.append("camera_id = ?")
        params.append(camera_id)
    if plate_text:
        conditions.append("plate_text LIKE ?")
        params.append(f"%{plate_text}%")

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Count total
    total = conn.execute(
        f"SELECT COUNT(*) as cnt FROM violations {where}", params
    ).fetchone()["cnt"]

    # Fetch page
    offset = (page - 1) * per_page
    rows = conn.execute(
        f"""SELECT * FROM violations {where}
            ORDER BY captured_at DESC
            LIMIT ? OFFSET ?""",
        params + [per_page, offset],
    ).fetchall()

    violations = []
    for r in rows:
        violations.append({
            "eventId": r["event_id"],
            "cameraId": r["camera_id"],
            "capturedAt": r["captured_at"],
            "evidence": {
                "imageOriginalUrl": r["image_original_url"],
                "imageEnhancedUrl": r["image_enhanced_url"],
                "videoClipUrl": r["video_clip_url"],
            },
            "violation": {
                "type": r["violation_type"],
                "measured": r["measured_speed"],
                "limit": r["speed_limit"],
            },
            "vehicle": {
                "vehicleClass": r["vehicle_class"],
                "plate": {
                    "text": r["plate_text"],
                    "confidence": r["plate_confidence"],
                },
            },
            "createdAt": r["created_at"],
        })

    pages = max(1, (total + per_page - 1) // per_page)

    return {
        "violations": violations,
        "total": total,
        "page": page,
        "perPage": per_page,
        "pages": pages,
    }


def get_violation_by_id(event_id):
    """Get a single violation by event ID."""
    conn = _get_conn()
    r = conn.execute(
        "SELECT * FROM violations WHERE event_id = ?", (event_id,)
    ).fetchone()
    if not r:
        return None
    return {
        "eventId": r["event_id"],
        "cameraId": r["camera_id"],
        "capturedAt": r["captured_at"],
        "evidence": {
            "imageOriginalUrl": r["image_original_url"],
            "imageEnhancedUrl": r["image_enhanced_url"],
            "videoClipUrl": r["video_clip_url"],
        },
        "violation": {
            "type": r["violation_type"],
            "measured": r["measured_speed"],
            "limit": r["speed_limit"],
        },
        "vehicle": {
            "vehicleClass": r["vehicle_class"],
            "plate": {
                "text": r["plate_text"],
                "confidence": r["plate_confidence"],
            },
        },
        "createdAt": r["created_at"],
    }


def get_violation_stats():
    """Get aggregate violation statistics."""
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) as cnt FROM violations").fetchone()["cnt"]
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM violations WHERE captured_at LIKE ?",
        (f"{today}%",),
    ).fetchone()["cnt"]

    avg_speed = conn.execute(
        "SELECT AVG(measured_speed) as avg_speed FROM violations"
    ).fetchone()["avg_speed"]

    max_speed = conn.execute(
        "SELECT MAX(measured_speed) as max_speed FROM violations"
    ).fetchone()["max_speed"]

    by_type = conn.execute(
        "SELECT vehicle_class, COUNT(*) as cnt FROM violations GROUP BY vehicle_class ORDER BY cnt DESC"
    ).fetchall()

    return {
        "totalViolations": total,
        "todayViolations": today_count,
        "avgSpeed": round(avg_speed, 1) if avg_speed else 0,
        "maxSpeed": round(max_speed, 1) if max_speed else 0,
        "byVehicleClass": {r["vehicle_class"] or "UNKNOWN": r["cnt"] for r in by_type},
    }


def delete_violation(event_id):
    """Delete a violation by event ID."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM violations WHERE event_id = ?", (event_id,))
    conn.commit()
    return cur.rowcount > 0
