"""Sky Robot: button driving, gripper, full-frame field vision and HSV tabs.

The ESP32 protocol is the one used by app2.py: F/B/L/R/S, SO/SC,
T followed by four speeds and newline, SA followed by four angles and newline,
and Q. The only driving interface is the on-screen buttons.
"""
import atexit
import ipaddress
import os
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from auto_approach import ApproachRunner
from vision_panel import VisionPanel


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
vision = VisionPanel(BASE_DIR / 'hsv_colour_config.json', BASE_DIR / 'field_config.json')
DEMO_MODE = bool(os.environ.get('FIELD_IMAGE'))
ESP32_IP = os.environ.get('ESP32_IP', '10.44.166.203')
ESP32_PORT = int(os.environ.get('ESP32_PORT', '80'))
ANGLE_KEYS = ('left_open', 'left_close', 'right_open', 'right_close')
SPEED_KEYS = ('M1_SPEED', 'M2_SPEED', 'LEFT_INNER_SPEED', 'RIGHT_INNER_SPEED')
speeds = dict(zip(SPEED_KEYS, (227, 245, 210, 210)))


class CarLink:
    def __init__(self, host, port):
        self.lock = threading.RLock()
        self.sock = None
        self.host = host
        self.port = port

    @property
    def status(self):
        with self.lock:
            return 'CONNECTED' if self.sock is not None else 'DISCONNECTED'

    @property
    def endpoint(self):
        with self.lock:
            return self.host, self.port

    def configure(self, host, port):
        with self.lock:
            self.close()
            self.host = host
            self.port = port

    def _connect(self):
        if DEMO_MODE:
            raise ConnectionError('โหมดภาพตัวอย่างปิดคำสั่งรถ')
        if self.sock is None:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=2)
                self.sock.settimeout(2)
            except OSError as exc:
                self.sock = None
                raise ConnectionError('เชื่อมต่อ ESP32 ไม่ได้') from exc

    def close(self):
        with self.lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def send(self, packet):
        with self.lock:
            self._connect()
            try:
                self.sock.sendall(packet.encode('ascii'))
            except OSError as exc:
                self.close()
                raise ConnectionError('ส่งคำสั่งไป ESP32 ไม่ได้') from exc

    def stop_if_connected(self):
        with self.lock:
            if self.sock is not None:
                try:
                    self.send('S')
                except ConnectionError:
                    pass

    def exchange(self, packet):
        with self.lock:
            self.send(packet)
            try:
                reply = bytearray()
                while len(reply) < 80 and not reply.endswith(b'\n'):
                    part = self.sock.recv(1)
                    if not part:
                        raise ConnectionError('ESP32 ตัดการเชื่อมต่อ')
                    reply.extend(part)
                if not reply.endswith(b'\n'):
                    raise ValueError('คำตอบจาก ESP32 ไม่ถูกต้อง')
                return reply.decode('ascii').strip()
            except (OSError, UnicodeError, ValueError, ConnectionError) as exc:
                self.close()
                raise ConnectionError('อ่านคำตอบจาก ESP32 ไม่ได้') from exc


link = CarLink(ESP32_IP, ESP32_PORT)
control_lock = threading.RLock()
button_active = False
button_moving = False
last_button_seen = 0.0
last_browser_seen = time.monotonic()


def touch_browser():
    global last_browser_seen
    last_browser_seen = time.monotonic()


field_lock = threading.RLock()
field_frame_lock = threading.Lock()
field_cap = None
field_running = False
field_thread = None
field_frame = None
mask_frame = None
field_error = None


def auto_ready():
    return (field_running and vision.mode == 'view' and not DEMO_MODE
            and not button_active and time.monotonic()-last_browser_seen < 3)


def auto_send(code):
    with control_lock:
        if auto.cancel.is_set() or not auto_ready():
            raise ValueError('ยกเลิก Auto หรือกล้องไม่พร้อม')
        link.send(code)


def auto_stop_motors():
    with control_lock:
        link.stop_if_connected()


auto = ApproachRunner(vision.auto_snapshot, auto_send, auto_stop_motors,
                      auto_ready)


class StillImageCamera:
    """Exercise the entire interface with a field photo and no car commands."""
    def __init__(self, path):
        try:
            self.image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            self.image = None

    def isOpened(self):
        return self.image is not None

    def read(self):
        time.sleep(.15)
        return self.image is not None, self.image.copy() if self.image is not None else None

    def release(self):
        pass


def field_worker():
    global field_frame, mask_frame, field_error, field_running, field_cap
    try:
        while field_running:
            with field_lock:
                if field_cap is None:
                    break
                ok, frame = field_cap.read()
            if not ok:
                time.sleep(.05)
                continue
            if frame.shape[1] > 1200:
                ratio = 1200 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=ratio, fy=ratio,
                                   interpolation=cv2.INTER_AREA)
            vision.set_frame(frame)
            shown = vision.render()
            mask = vision.colour_mask()
            encoded_ok, jpg = cv2.imencode('.jpg', shown,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
            mask_ok, mask_jpg = (cv2.imencode('.jpg', mask, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                 if mask is not None else (False, None))
            if encoded_ok:
                with field_frame_lock:
                    field_frame = jpg.tobytes()
                    mask_frame = mask_jpg.tobytes() if mask_ok else None
    except (cv2.error, ValueError) as exc:
        field_error = str(exc)
    finally:
        with field_lock:
            field_running = False
            if field_cap is not None:
                field_cap.release()
                field_cap = None
        with field_frame_lock:
            field_frame = None
            mask_frame = None


def start_field_camera():
    global field_cap, field_running, field_thread, field_error
    with field_lock:
        if field_running:
            return True
        example = os.environ.get('FIELD_IMAGE')
        camera = (StillImageCamera(example) if example else
                  cv2.VideoCapture(int(os.environ.get('FIELD_CAMERA', '1'))))
        if not camera.isOpened():
            camera.release()
            field_error = 'เปิดกล้องสนามไม่ได้ ตรวจหมายเลขกล้องหรือไฟล์ภาพ'
            return False
        if not example:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,
                       int(os.environ.get('FIELD_WIDTH', '1920')))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                       int(os.environ.get('FIELD_HEIGHT', '1080')))
        field_error = None
        field_cap = camera
        field_running = True
    field_thread = threading.Thread(target=field_worker, daemon=True)
    field_thread.start()
    return True


def stop_field_camera():
    global field_running, field_thread, field_frame, mask_frame, field_cap
    if auto.state()['running']:
        auto.abort()
    with field_lock:
        field_running = False
        if field_cap is not None:
            field_cap.release()
            field_cap = None
    with field_frame_lock:
        field_frame = None
        mask_frame = None
    if field_thread is not None and field_thread is not threading.current_thread():
        field_thread.join(timeout=3)
        field_thread = None


def generate_field_frames(mask_only=False):
    while field_running:
        touch_browser()
        with field_frame_lock:
            data = mask_frame if mask_only else field_frame
        if data is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        time.sleep(.05)


@app.route('/')
def index():
    touch_browser()
    return render_template('index.html')


@app.route('/field/start', methods=['POST'])
def field_start():
    touch_browser()
    ok = start_field_camera()
    return jsonify(ok=ok, field='ON' if ok else 'OFF',
                   error=None if ok else field_error), 200 if ok else 503


@app.route('/field/stop', methods=['POST'])
def field_stop():
    stop_field_camera()
    return jsonify(ok=True, field='OFF')


@app.route('/auto/start', methods=['POST'])
def auto_start():
    touch_browser()
    try:
        auto.start()
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 409
    return jsonify(ok=True, **auto.state())


@app.route('/auto/stop', methods=['POST'])
def auto_stop():
    auto.abort()
    return jsonify(ok=True, **auto.state())


@app.route('/field/video_feed')
def field_video_feed():
    if not field_running:
        return ('Field camera off', 409)
    return Response(generate_field_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/field/mask_feed')
def field_mask_feed():
    if not field_running:
        return ('Field camera off', 409)
    return Response(generate_field_frames(mask_only=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vision/state')
def vision_state():
    touch_browser()
    return jsonify(ok=True, **vision.state())


@app.route('/vision/action', methods=['POST'])
def vision_action():
    global button_active, button_moving
    touch_browser()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error='ข้อมูลไม่ถูกต้อง'), 400
    action = data.get('action')
    try:
        if action == 'mode':
            mode = data.get('mode')
            if mode not in ('view', 'colour', 'arena', 'hole'):
                raise ValueError('โหมดไม่ถูกต้อง')
            if auto.state()['running']:
                auto.abort()
            with control_lock:
                link.stop_if_connected()
                button_active = False
                button_moving = False
                vision.set_mode(mode)
        elif action == 'select':
            vision.select(data.get('colour'), hole=data.get('hole') is True)
        elif action == 'sliders':
            if vision.mode != 'colour':
                raise ValueError('เปิดแท็บปรับสี HSV ก่อน')
            vision.sliders(data.get('values'))
        elif action == 'click':
            vision.click(data.get('point'))
        elif action == 'undo_sample':
            vision.undo()
        elif action == 'reset_colour':
            vision.reset_colour()
        elif action == 'save_colours':
            vision.save_colours()
        elif action == 'reset_arena':
            vision.reset_arena()
        elif action == 'undo_arena':
            vision.undo_arena()
        elif action == 'clear_hole':
            vision.clear_hole()
        elif action == 'save_field':
            vision.save_field()
        else:
            raise ValueError('คำสั่งปรับภาพไม่ถูกต้อง')
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, **vision.state())


@app.route('/manual/open', methods=['POST'])
def manual_open():
    global button_active, button_moving, last_button_seen
    touch_browser()
    if vision.mode != 'view' or DEMO_MODE:
        return jsonify(ok=False, error='กลับแท็บควบคุมรถก่อน หรือปิดโหมดภาพตัวอย่าง'), 409
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนเปิดปุ่มควบคุม'), 409
    with control_lock:
        link.stop_if_connected()
        button_active = True
        button_moving = False
        last_button_seen = time.monotonic()
    return jsonify(ok=True)


@app.route('/manual/drive', methods=['POST'])
def manual_drive():
    global last_button_seen, button_active, button_moving
    touch_browser()
    data = request.get_json(silent=True)
    code = data.get('code') if isinstance(data, dict) else None
    if code not in ('F', 'B', 'L', 'R', 'S'):
        return jsonify(ok=False, error='คำสั่งรถไม่ถูกต้อง'), 400
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนใช้ปุ่มขับรถ'), 409
    with control_lock:
        if not button_active or vision.mode != 'view' or DEMO_MODE:
            return jsonify(ok=False, error='เปิดโหมดควบคุมด้วยปุ่มก่อน'), 409
        last_button_seen = time.monotonic()
        try:
            link.send(code)
            button_moving = code != 'S'
        except ConnectionError as exc:
            button_active = False
            button_moving = False
            return jsonify(ok=False, error=str(exc)), 503
    return jsonify(ok=True)


@app.route('/manual/close', methods=['POST'])
def manual_close():
    global button_active, button_moving
    if auto.state()['running']:
        auto.abort()
    with control_lock:
        link.stop_if_connected()
        button_active = False
        button_moving = False
    return jsonify(ok=True)


@app.route('/gripper', methods=['POST'])
def gripper_control():
    global button_moving
    data = request.get_json(silent=True)
    action = data.get('action') if isinstance(data, dict) else None
    if action not in ('open', 'close'):
        return jsonify(ok=False, error='คำสั่ง Gripper ไม่ถูกต้อง'), 400
    if vision.mode != 'view' or DEMO_MODE:
        return jsonify(ok=False, error='กลับแท็บควบคุมรถก่อน'), 409
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนใช้ Gripper'), 409
    with control_lock:
        button_moving = False
        link.stop_if_connected()
        try:
            link.send('SO' if action == 'open' else 'SC')
        except ConnectionError as exc:
            return jsonify(ok=False, error=str(exc)), 503
    return jsonify(ok=True)


@app.route('/speeds', methods=['GET', 'POST'])
def speed_settings():
    if request.method == 'GET':
        return jsonify(speeds)
    if vision.mode != 'view' or DEMO_MODE:
        return jsonify(ok=False, error='กลับแท็บควบคุมรถก่อน'), 409
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนปรับความเร็ว'), 409
    values = request.get_json(silent=True)
    if (not isinstance(values, dict) or set(values) != set(SPEED_KEYS) or
            any(type(values[k]) is not int or not 0 <= values[k] <= 255 for k in SPEED_KEYS)):
        return jsonify(ok=False, error='ความเร็วทั้ง 4 ค่าเป็นจำนวนเต็ม 0–255'), 400
    with control_lock:
        global button_moving
        button_moving = False
        link.stop_if_connected()
        try:
            answer = link.exchange('T' + ','.join(str(values[k]) for k in SPEED_KEYS) + '\n')
            if answer != 'OK':
                raise ValueError('ESP32 ไม่ยอมรับค่าความเร็ว')
            speeds.update(values)
        except (ConnectionError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 503
    return jsonify(ok=True, speeds=speeds)


@app.route('/gripper/angles', methods=['GET', 'POST'])
def gripper_angles():
    if DEMO_MODE or vision.mode != 'view':
        return jsonify(ok=False, error='กลับแท็บควบคุมรถก่อน'), 409
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนปรับมุม'), 409
    values = request.get_json(silent=True) if request.method == 'POST' else None
    if request.method == 'POST' and (not isinstance(values, dict) or
            set(values) != set(ANGLE_KEYS) or
            any(type(values[k]) is not int or not 0 <= values[k] <= 180 for k in ANGLE_KEYS)):
        return jsonify(ok=False, error='มุมทั้ง 4 ค่าเป็นจำนวนเต็ม 0–180'), 400
    with control_lock:
        global button_moving
        button_moving = False
        link.stop_if_connected()
        try:
            if values is not None:
                packet = 'SA' + ','.join(str(values[k]) for k in ANGLE_KEYS) + '\n'
                if link.exchange(packet) != 'OK':
                    raise ValueError('ESP32 ไม่ยอมรับค่ามุม')
            parts = link.exchange('Q').split(',')
            if len(parts) != 5 or parts[0] != 'ANGLES':
                raise ValueError('ESP32 ตอบค่ามุมไม่ถูกต้อง')
            actual = dict(zip(ANGLE_KEYS, (int(n) for n in parts[1:])))
            if any(not 0 <= n <= 180 for n in actual.values()):
                raise ValueError('ค่ามุมจาก ESP32 ไม่ถูกต้อง')
            if values is not None and actual != values:
                raise ValueError('ค่าที่อ่านกลับไม่ตรงกับที่บันทึก')
        except (ConnectionError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 503
    return jsonify(ok=True, angles=actual)


@app.route('/reconnect', methods=['POST'])
def reconnect():
    if DEMO_MODE:
        return jsonify(ok=False, error='โหมดภาพตัวอย่างปิดการเชื่อมรถ'), 409
    if auto.state()['running']:
        return jsonify(ok=False, error='หยุด Auto ก่อนเชื่อมต่อใหม่'), 409
    data = request.get_json(silent=True)
    if data is not None and not isinstance(data, dict):
        return jsonify(ok=False, error='ข้อมูลการเชื่อมต่อไม่ถูกต้อง'), 400
    requested_ip = (data or {}).get('ip')
    if requested_ip is not None:
        if not isinstance(requested_ip, str):
            return jsonify(ok=False, error='IP ต้องเป็นข้อความ'), 400
        try:
            requested_ip = str(ipaddress.IPv4Address(requested_ip.strip()))
        except ipaddress.AddressValueError:
            return jsonify(ok=False, error='รูปแบบ IPv4 ไม่ถูกต้อง'), 400
    with control_lock:
        link.stop_if_connected()
        if requested_ip is not None:
            link.configure(requested_ip, ESP32_PORT)
        else:
            link.close()
        try:
            link.send('S')
        except ConnectionError as exc:
            return jsonify(ok=False, error=str(exc)), 503
    host, port = link.endpoint
    return jsonify(ok=True, esp32=link.status, esp32_ip=host, esp32_port=port)


@app.route('/status')
def status():
    touch_browser()
    host, port = link.endpoint
    return jsonify(esp32=link.status, field='ON' if field_running else 'OFF',
                   field_error=field_error, manual=button_active, demo=DEMO_MODE,
                   auto=auto.state(), esp32_ip=host, esp32_port=port)


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    touch_browser()
    return jsonify(ok=True)


@app.route('/page_closed', methods=['POST'])
def page_closed():
    auto.abort()
    manual_close()
    stop_field_camera()
    return ('', 204)


def watchdog():
    global button_active, button_moving
    while True:
        time.sleep(.15)
        if button_moving and time.monotonic() - last_button_seen > .65:
            with control_lock:
                if button_moving and time.monotonic() - last_button_seen > .65:
                    link.stop_if_connected()
                    button_moving = False
        if time.monotonic() - last_browser_seen > 3:
            if auto.state()['running']:
                auto.abort()
            with control_lock:
                if button_active:
                    link.stop_if_connected()
                    button_active = False
                    button_moving = False
            if field_running:
                stop_field_camera()


def cleanup():
    auto.abort()
    stop_field_camera()
    link.stop_if_connected()
    link.close()


atexit.register(cleanup)
threading.Thread(target=watchdog, daemon=True).start()

if __name__ == '__main__':
    print('Open http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
