"""
WebSocket RC Car — zero latency
Run on Raspberry Pi: python3 web_control.py /dev/ttyACM0
Open http://<raspberrypi-ip>:5000 in browser to control
"""

import sys
import time
import threading
import io

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from car_controller import MotorController

# ============================================
# Camera
# ============================================
camera = None
camera_lock = threading.Lock()
CAM_WIDTH  = 480
CAM_HEIGHT = 360

def init_camera():
    """Initialize camera: try picamera2 first, fallback to OpenCV"""
    global camera
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        print(f"[Camera] picamera2 started ({CAM_WIDTH}x{CAM_HEIGHT})", flush=True)
        camera = ('picamera2', picam2)
        return
    except Exception as e:
        print(f"[Camera] picamera2 failed: {e}", flush=True)

    try:
        import cv2
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 15)
                print(f"[Camera] OpenCV /dev/video{idx} ({CAM_WIDTH}x{CAM_HEIGHT})", flush=True)
                camera = ('opencv', cap)
                return
            cap.release()
    except Exception as e:
        print(f"[Camera] OpenCV failed: {e}", flush=True)

    print("[Camera] No camera found", flush=True)

def read_frame():
    """Read one frame, returns numpy BGR image or None"""
    if camera is None:
        return None
    cam_type, cam_obj = camera
    if cam_type == 'picamera2':
        import numpy as np
        return cam_obj.capture_array()
    elif cam_type == 'opencv':
        ret, frame = cam_obj.read()
        return frame if ret else None
    return None

def release_camera():
    global camera
    if camera is None:
        return
    cam_type, cam_obj = camera
    try:
        if cam_type == 'picamera2':
            cam_obj.stop()
        elif cam_type == 'opencv':
            cam_obj.release()
    except Exception:
        pass
    camera = None
    print("[Camera] Released", flush=True)

# ============================================
# Configuration
# ============================================
GEAR_PRESETS = {1: 130, 2: 170, 3: 210, 4: 255}
DEFAULT_GEAR = 3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'car-remote-2024'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Global motor controller
mc = None
mc_lock = threading.Lock()
obstacle_alert = False
obstacle_enabled = True  # obstacle avoidance toggle state

# ============================================
# Routes
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """MJPEG stream generator"""
    import cv2
    while True:
        with camera_lock:
            frame = read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~15-20 fps

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ============================================
# WebSocket event handlers
# ============================================
@socketio.on('connect')
def on_connect():
    global obstacle_alert
    print(f"[Connect] Client connected", flush=True)
    emit('status', {
        'gear': DEFAULT_GEAR,
        'speed': GEAR_PRESETS[DEFAULT_GEAR],
        'obstacle_alert': obstacle_alert,
        'distance': -1,
    })

@socketio.on('disconnect')
def on_disconnect():
    print("[Disconnect] Client left", flush=True)
    with mc_lock:
        if mc:
            mc.car_stop()

@socketio.on('command')
def on_command(data):
    """Handle remote control commands"""
    global obstacle_enabled, obstacle_alert
    cmd = data.get('cmd', '')
    print(f"[Command] {cmd}", flush=True)

    with mc_lock:
        if not mc:
            emit('error', {'msg': 'Arduino not connected'})
            return

        # Movement commands
        if cmd == 'FORWARD':
            mc.car_forward()
        elif cmd == 'BACKWARD':
            mc.car_backward()
        elif cmd == 'LEFT':
            mc.car_left()
        elif cmd == 'RIGHT':
            mc.car_right()
        elif cmd == 'CLEFT':
            mc.car_curve_left()
        elif cmd == 'CRIGHT':
            mc.car_curve_right()
        elif cmd == 'CBLEFT':
            mc.car_curve_back_left()
        elif cmd == 'CBRIGHT':
            mc.car_curve_back_right()
        elif cmd == 'STOP':
            mc.car_stop()

        # Speed control
        elif cmd == 'SET_SPEED':
            gear = data.get('gear', DEFAULT_GEAR)
            if gear in GEAR_PRESETS:
                speed = GEAR_PRESETS[gear]
                mc.car_set_speed(speed)
                emit('status', {
                    'gear': gear,
                    'speed': speed,
                    'obstacle_alert': obstacle_alert,
                })

        # Distance query
        elif cmd == 'DISTANCE':
            d = mc.car_read_distance()
            emit('distance', {'value': d})

        # Obstacle avoidance toggle
        elif cmd == 'OBSTACLE_ON':
            mc.car_obstacle_on()
            obstacle_enabled = True
            emit('obstacle_toggle', {'enabled': True})

        elif cmd == 'OBSTACLE_OFF':
            mc.car_obstacle_off()
            obstacle_enabled = False
            obstacle_alert = False
            emit('obstacle_toggle', {'enabled': False})

# ============================================
# Background thread: Arduino serial listener + distance polling
# ============================================
def arduino_listener():
    """Continuously read async messages from Arduino (obstacle alerts, etc.)"""
    global obstacle_alert
    last_dist_query = 0

    while True:
        with mc_lock:
            m = mc
        if m is None:
            time.sleep(0.5)
            continue

        now = time.time()

        # 1. Non-blocking read from Arduino
        try:
            msgs = m.read_pending()
            for msg in msgs:
                mtype = msg[0]
                if mtype == 'OBSTACLE':
                    obstacle_alert = True
                    dist = msg[1] if msg[1] else 0
                    turning = len(msg) > 2 and msg[2] == 'TURNING'
                    socketio.emit('obstacle', {
                        'distance': dist,
                        'turning': turning,
                    })
                elif mtype == 'AVOID_DONE':
                    obstacle_alert = False
                    socketio.emit('avoid_done', {})
                elif mtype == 'DIST':
                    socketio.emit('distance', {'value': msg[1]})
        except Exception:
            pass

        # 2. Poll distance every 1 second
        if now - last_dist_query > 1.0:
            try:
                m.ser.write(b"DISTANCE\n")
                last_dist_query = now
            except Exception:
                pass

        time.sleep(0.1)

# ============================================
# Entry point
# ============================================
def main():
    global mc

    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'

    print("=" * 55)
    print("  WebSocket RC Car Server")
    print("=" * 55)
    print(f"Connecting to Arduino ({port})...", flush=True)

    mc = MotorController(port=port)
    mc.car_set_speed(GEAR_PRESETS[DEFAULT_GEAR])
    print(f"Connected! Gear: {DEFAULT_GEAR} (speed={GEAR_PRESETS[DEFAULT_GEAR]})")
    print()

    # Initialize camera
    init_camera()

    # Start background listener thread
    socketio.start_background_task(arduino_listener)

    # Start web server
    print("Starting web server...")
    print("Open in browser: http://<raspberrypi-ip>:5000")
    print()
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if mc:
            mc.car_stop()
            mc.close()
        release_camera()
        print("Bye")

if __name__ == '__main__':
    main()
