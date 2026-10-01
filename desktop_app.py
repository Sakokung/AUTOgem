"""Native desktop UI for camera, vision and ESP32 control (no web server)."""
from __future__ import annotations

import io
import ipaddress
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

import app as backend
from vision_panel import KEYS, LIMITS


COLOURS = ('BLUE', 'CYAN', 'GREEN', 'PURPLE', 'ORANGE', 'RED')


class ImageCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, background='#111827', highlightthickness=0, **kwargs)
        self.photo = self.image_box = self._jpeg = None

    def show_jpeg(self, jpeg):
        if not jpeg or jpeg is self._jpeg:
            return
        self._jpeg = jpeg
        try:
            image = Image.open(io.BytesIO(jpeg)).convert('RGB')
        except (OSError, ValueError):
            return
        cw, ch = max(2, self.winfo_width()), max(2, self.winfo_height())
        scale = min(cw / image.width, ch / image.height)
        width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        x, y = (cw - width) // 2, (ch - height) // 2
        self.delete('all')
        self.create_image(x, y, anchor='nw', image=self.photo)
        self.image_box = (x, y, width, height)

    def normalized_point(self, event):
        if self.image_box is None:
            return None
        x, y, width, height = self.image_box
        if x <= event.x < x + width and y <= event.y < y + height:
            return [(event.x-x)/width, (event.y-y)/height]
        return None

    def clear_image(self):
        if self._jpeg is None and self.find_all():
            return
        self._jpeg = self.photo = self.image_box = None
        self.delete('all')
        self.create_text(max(1, self.winfo_width())//2, max(1, self.winfo_height())//2,
                         text='กล้องปิดอยู่', fill='#94a3b8', font=('TkDefaultFont', 14))


class DesktopApp:
    def __init__(self, root):
        self.root = root
        root.title('SKY ROBOT — Local Control')
        root.geometry('1400x850')
        root.minsize(1050, 680)
        self.closing = self.manual = False
        self.drive_code = self.drive_job = self.slider_job = None
        self.status_var = tk.StringVar(value='กำลังเริ่มระบบ…')
        self.message_var = tk.StringVar(value='พร้อม')
        self._style()
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self._tick)
        root.after(250, self.start_camera)

    def _style(self):
        style = ttk.Style(self.root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('Title.TLabel', font=('TkDefaultFont', 15, 'bold'))
        style.configure('Section.TLabel', font=('TkDefaultFont', 11, 'bold'))
        style.configure('Danger.TButton', foreground='#b91c1c')

    def _build(self):
        header = ttk.Frame(self.root, padding=(12, 8)); header.pack(fill='x')
        ttk.Label(header, text='SKY ROBOT  •  Local Desktop', style='Title.TLabel').pack(side='left')
        ttk.Label(header, textvariable=self.status_var).pack(side='right')
        camera = ttk.Frame(self.root, padding=(12, 0, 12, 8)); camera.pack(fill='x')
        ttk.Label(camera, text='กล้อง:').pack(side='left')
        self.camera_source = tk.StringVar(value=backend.field_camera_source)
        source = ttk.Combobox(camera, textvariable=self.camera_source, width=20,
                              values=('iriun', 'usb'), state='readonly')
        source.pack(side='left', padx=5); source.bind('<<ComboboxSelected>>', self._camera_default)
        ttk.Label(camera, text='Camera Index').pack(side='left', padx=(10, 3))
        self.camera_index = tk.IntVar(value=backend.field_camera_index)
        ttk.Spinbox(camera, from_=0, to=99, textvariable=self.camera_index, width=5).pack(side='left')
        ttk.Button(camera, text='เปิดกล้อง', command=self.start_camera).pack(side='left', padx=(10, 4))
        ttk.Button(camera, text='ปิดกล้อง', command=self.stop_camera).pack(side='left')
        ttk.Label(camera, textvariable=self.message_var).pack(side='left', padx=14)

        body = ttk.Panedwindow(self.root, orient='horizontal'); body.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        pictures, controls = ttk.Frame(body), ttk.Frame(body, width=440)
        body.add(pictures, weight=3); body.add(controls, weight=2)
        self.image_canvas = ImageCanvas(pictures, height=560); self.image_canvas.pack(fill='both', expand=True)
        self.image_canvas.bind('<Button-1>', self.image_clicked)
        self.mask_frame = ttk.LabelFrame(pictures, text='HSV Mask Preview', padding=4)
        self.mask_canvas = ImageCanvas(self.mask_frame, height=210); self.mask_canvas.pack(fill='both', expand=True)
        self.tabs = ttk.Notebook(controls); self.tabs.pack(fill='both', expand=True)
        self.control_tab, self.colour_tab, self.field_tab = (ttk.Frame(self.tabs, padding=12) for _ in range(3))
        self.tabs.add(self.control_tab, text='ควบคุมรถ'); self.tabs.add(self.colour_tab, text='ปรับสี HSV'); self.tabs.add(self.field_tab, text='ตั้งสนาม')
        self.tabs.bind('<<NotebookTabChanged>>', self.tab_changed)
        self._build_control(); self._build_colour(); self._build_field()

    def _build_control(self):
        tab = self.control_tab
        box = ttk.LabelFrame(tab, text='การเชื่อมต่อ ESP32', padding=8); box.pack(fill='x')
        self.ip_var = tk.StringVar(value=backend.link.endpoint[0])
        ttk.Entry(box, textvariable=self.ip_var, width=18).pack(side='left', fill='x', expand=True)
        ttk.Button(box, text='เชื่อมต่อ', command=self.reconnect).pack(side='left', padx=(6, 0))
        manual = ttk.LabelFrame(tab, text='ขับรถด้วยปุ่ม', padding=8); manual.pack(fill='x', pady=8)
        top = ttk.Frame(manual); top.pack()
        ttk.Button(top, text='เปิดใช้ปุ่ม', command=self.open_manual).pack(side='left', padx=3)
        ttk.Button(top, text='ปิดใช้ปุ่ม', command=self.close_manual).pack(side='left', padx=3)
        pad = ttk.Frame(manual); pad.pack(pady=8)
        for text, code, row, col in [('หน้า','F',0,1),('ซ้าย','L',1,0),('หยุด','S',1,1),('ขวา','R',1,2),('ถอย','B',2,1)]:
            self._drive_button(pad, text, code, row, col)
        row = ttk.Frame(manual); row.pack(fill='x')
        for text, command in [('เปิด Gripper',lambda:self.gripper('open')),('ปิด Gripper',lambda:self.gripper('close')),('หยุดฉุกเฉิน',self.emergency)]:
            ttk.Button(row, text=text, command=command).pack(side='left', expand=True, fill='x', padx=2)
        auto = ttk.LabelFrame(tab, text='Auto', padding=8); auto.pack(fill='x', pady=(0,8))
        ttk.Button(auto, text='เริ่ม Auto', command=self.start_auto).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(auto, text='หยุด Auto', command=self.stop_auto).pack(side='left', expand=True, fill='x', padx=2)
        speed = ttk.LabelFrame(tab, text='ความเร็วล้อ (0–255)', padding=8); speed.pack(fill='x')
        self.speed_vars = {}
        for i, key in enumerate(backend.SPEED_KEYS):
            ttk.Label(speed, text=key).grid(row=i, column=0, sticky='w')
            self.speed_vars[key] = tk.IntVar(value=backend.speeds[key])
            ttk.Spinbox(speed, from_=0, to=255, textvariable=self.speed_vars[key], width=7).grid(row=i, column=1, padx=5, pady=2)
        ttk.Button(speed, text='ส่งค่าความเร็ว', command=self.save_speeds).grid(row=0, column=2, rowspan=2, padx=8)
        angles = ttk.LabelFrame(tab, text='มุม Servo (0–180)', padding=8); angles.pack(fill='x', pady=(8,0))
        self.angle_vars = {}
        for i, (key, default) in enumerate(zip(backend.ANGLE_KEYS, (50,120,130,60))):
            ttk.Label(angles, text=key).grid(row=i, column=0, sticky='w')
            self.angle_vars[key] = tk.IntVar(value=default)
            ttk.Spinbox(angles, from_=0, to=180, textvariable=self.angle_vars[key], width=7).grid(row=i, column=1, padx=5, pady=2)
        ttk.Button(angles, text='อ่านค่า', command=self.load_angles).grid(row=0, column=2, padx=5)
        ttk.Button(angles, text='บันทึก', command=self.save_angles).grid(row=1, column=2, padx=5)

    def _drive_button(self, parent, text, code, row, col):
        button = ttk.Button(parent, text=text, width=11); button.grid(row=row, column=col, padx=3, pady=3)
        button.bind('<ButtonPress-1>', lambda _e: self.drive_start(code))
        button.bind('<ButtonRelease-1>', lambda _e: self.drive_stop())
        button.bind('<Leave>', lambda _e: self.drive_stop())

    def _build_colour(self):
        tab = self.colour_tab
        ttk.Label(tab, text='เลือกสีแล้วคลิกกลางหินในภาพ', style='Section.TLabel').pack(anchor='w')
        self.colour_var = tk.StringVar(value=backend.vision.colour)
        combo = ttk.Combobox(tab, textvariable=self.colour_var, values=COLOURS, state='readonly')
        combo.pack(fill='x', pady=7); combo.bind('<<ComboboxSelected>>', self.select_colour)
        sliders = ttk.Frame(tab); sliders.pack(fill='both', expand=True)
        self.slider_vars = {}
        for row, (key, maximum) in enumerate(zip(KEYS, LIMITS)):
            ttk.Label(sliders, text=key, width=14).grid(row=row, column=0, sticky='w')
            var = self.slider_vars[key] = tk.IntVar()
            ttk.Scale(sliders, from_=0, to=maximum, variable=var, command=lambda _v:self.slider_changed()).grid(row=row, column=1, sticky='ew', padx=4)
            ttk.Label(sliders, textvariable=var, width=5).grid(row=row, column=2)
        sliders.columnconfigure(1, weight=1)
        row = ttk.Frame(tab); row.pack(fill='x', pady=8)
        ttk.Button(row, text='ย้อนจุ่ม', command=lambda:self.vision_action(backend.vision.undo)).pack(side='left', padx=2)
        ttk.Button(row, text='คืนค่าเริ่มต้น', command=lambda:self.vision_action(backend.vision.reset_colour)).pack(side='left', padx=2)
        ttk.Button(row, text='บันทึก HSV JSON', command=lambda:self.vision_action(backend.vision.save_colours, 'บันทึก hsv_colour_config.json แล้ว')).pack(side='left', padx=2)
        self.sync_sliders()

    def _build_field(self):
        tab = self.field_tab
        ttk.Label(tab, text='ตั้งกรอบสนามและวงสีจากภาพเต็ม', style='Section.TLabel').pack(anchor='w')
        ttk.Label(tab, text='กรอบ: คลิก 4 มุมเรียงรอบสนาม\nวงสี: คลิกกลางวง แล้วคลิกขอบวง').pack(anchor='w', pady=6)
        row = ttk.Frame(tab); row.pack(fill='x', pady=5)
        ttk.Button(row, text='โหมด 4 มุม', command=lambda:self.set_mode('arena')).pack(side='left', padx=2)
        ttk.Button(row, text='ย้อนมุม', command=lambda:self.vision_action(backend.vision.undo_arena)).pack(side='left', padx=2)
        ttk.Button(row, text='เริ่มกรอบใหม่', command=lambda:self.vision_action(backend.vision.reset_arena)).pack(side='left', padx=2)
        row = ttk.Frame(tab); row.pack(fill='x', pady=5)
        ttk.Button(row, text='โหมดวงสี', command=lambda:self.set_mode('hole')).pack(side='left', padx=2)
        self.hole_var = tk.StringVar(value=backend.vision.hole_colour)
        combo = ttk.Combobox(row, textvariable=self.hole_var, values=COLOURS, state='readonly', width=10)
        combo.pack(side='left', padx=3); combo.bind('<<ComboboxSelected>>', self.select_hole)
        ttk.Button(row, text='ลบวงนี้', command=lambda:self.vision_action(backend.vision.clear_hole)).pack(side='left', padx=2)
        ttk.Button(tab, text='บันทึก field_config.json', command=lambda:self.vision_action(backend.vision.save_field, 'บันทึก field_config.json แล้ว')).pack(fill='x', pady=8)
        self.field_progress = tk.StringVar(); ttk.Label(tab, textvariable=self.field_progress).pack(anchor='w')

    def run_background(self, operation, success=None):
        def worker():
            try: result = operation()
            except Exception as exc:
                if not self.closing: self.root.after(0, lambda e=exc:self.set_message(str(e), True))
            else:
                if success and not self.closing: self.root.after(0, lambda:success(result))
        threading.Thread(target=worker, daemon=True).start()

    def set_message(self, text, error=False):
        self.message_var.set(('ผิดพลาด: ' if error else '') + text)

    def _camera_default(self, _event=None):
        self.camera_index.set(backend.CAMERA_DEFAULTS[self.camera_source.get()])

    def start_camera(self):
        try:
            source, index = self.camera_source.get(), int(self.camera_index.get())
            if source not in backend.CAMERA_LABELS or not 0 <= index <= 99: raise ValueError
        except (ValueError, tk.TclError):
            self.set_message('Camera Index ต้องเป็น 0–99', True); return
        def operation():
            if backend.field_running and (source != backend.field_camera_source or index != backend.field_camera_index): backend.stop_field_camera()
            backend.field_camera_source, backend.field_camera_index = source, index
            if not backend.start_field_camera(): raise RuntimeError(backend.field_error or 'เปิดกล้องไม่ได้')
        self.set_message('กำลังเปิดกล้อง…'); self.run_background(operation, lambda _r:self.set_message('กล้องพร้อม'))

    def stop_camera(self):
        self.run_background(backend.stop_field_camera, lambda _r:self.set_message('ปิดกล้องแล้ว'))

    def tab_changed(self, _event=None):
        name = self.tabs.tab(self.tabs.select(), 'text')
        mode = 'view' if name == 'ควบคุมรถ' else 'colour' if name == 'ปรับสี HSV' else 'arena'
        self.mask_frame.pack(fill='both', pady=(6,0)) if mode == 'colour' else self.mask_frame.pack_forget()
        self.close_manual(); self.set_mode(mode)
        if mode == 'colour': self.sync_sliders()

    def set_mode(self, mode):
        backend.auto.abort(); backend.link.stop_if_connected(); backend.vision.set_mode(mode)

    def image_clicked(self, event):
        if backend.vision.mode == 'view': return
        point = self.image_canvas.normalized_point(event)
        if point is not None: self.vision_action(lambda:backend.vision.click(point), 'รับพิกัดแล้ว')

    def vision_action(self, operation, message='อัปเดตแล้ว'):
        try:
            operation(); self.set_message(message)
            if backend.vision.mode == 'colour': self.sync_sliders()
        except (ValueError, KeyError, TypeError) as exc: self.set_message(str(exc), True)

    def select_colour(self, _event=None): self.vision_action(lambda:backend.vision.select(self.colour_var.get()))
    def select_hole(self, _event=None): self.vision_action(lambda:backend.vision.select(self.hole_var.get(), hole=True))
    def sync_sliders(self):
        for key, value in backend.vision.state()['values'].items(): self.slider_vars[key].set(value)
    def slider_changed(self):
        if self.slider_job is not None: self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(180, self.apply_sliders)
    def apply_sliders(self):
        self.slider_job = None
        values = {key:int(var.get()) for key,var in self.slider_vars.items()}
        self.vision_action(lambda:backend.vision.sliders(values))

    def reconnect(self):
        try: host = str(ipaddress.IPv4Address(self.ip_var.get().strip()))
        except ipaddress.AddressValueError: self.set_message('รูปแบบ IPv4 ไม่ถูกต้อง', True); return
        def operation(): backend.link.configure(host, backend.ESP32_PORT); backend.link.send('S')
        self.run_background(operation, lambda _r:self.set_message(f'เชื่อมต่อ {host} แล้ว'))

    def open_manual(self):
        if backend.DEMO_MODE: self.set_message('โหมดภาพตัวอย่างปิดคำสั่งรถ', True); return
        if backend.auto.state()['running']: self.set_message('หยุด Auto ก่อน', True); return
        backend.link.stop_if_connected(); backend.button_active = True; backend.button_moving = False
        backend.last_button_seen = time.monotonic(); self.manual = True; self.set_message('พร้อมขับด้วยปุ่ม')
    def close_manual(self):
        self.drive_stop(); backend.link.stop_if_connected(); backend.button_active = backend.button_moving = False; self.manual = False
    def drive_start(self, code):
        if not self.manual: self.set_message('กดเปิดใช้ปุ่มก่อน', True); return
        self.drive_code = code; self._send_drive()
    def _send_drive(self):
        if self.drive_code is None or not self.manual: return
        code = self.drive_code
        def operation():
            # A queued network operation may begin after ButtonRelease.  Check
            # again both before and under the shared control lock so an old
            # movement packet can never overtake the final stop packet.
            if self.drive_code != code or not self.manual:
                return
            with backend.control_lock:
                if self.drive_code != code or not self.manual:
                    return
                backend.link.send(code); backend.last_button_seen = time.monotonic(); backend.button_moving = code != 'S'
        self.run_background(operation); self.drive_job = self.root.after(200, self._send_drive)
    def drive_stop(self):
        if self.drive_job is not None: self.root.after_cancel(self.drive_job); self.drive_job = None
        moving, self.drive_code = self.drive_code is not None, None
        if moving and self.manual: self.run_background(lambda:backend.link.send('S')); backend.button_moving = False
    def gripper(self, action):
        if backend.auto.state()['running']: self.set_message('หยุด Auto ก่อนใช้ Gripper', True); return
        self.run_background(lambda:backend.link.send('SO' if action=='open' else 'SC'), lambda _r:self.set_message('ส่งคำสั่ง Gripper แล้ว'))
    def emergency(self): backend.auto.abort(); self.close_manual(); backend.link.stop_if_connected(); self.set_message('หยุดรถแล้ว')
    def start_auto(self):
        self.close_manual()
        try: backend.auto.start(); self.set_message('เริ่ม Auto แล้ว')
        except ValueError as exc: self.set_message(str(exc), True)
    def stop_auto(self): backend.auto.abort(); self.set_message('หยุด Auto แล้ว')

    def save_speeds(self):
        if backend.auto.state()['running']:
            self.set_message('หยุด Auto ก่อนปรับความเร็ว', True); return
        try:
            values = {k:int(v.get()) for k,v in self.speed_vars.items()}
            if any(not 0 <= v <= 255 for v in values.values()): raise ValueError
        except (ValueError, tk.TclError): self.set_message('ความเร็วต้องเป็นจำนวนเต็ม 0–255', True); return
        packet = 'T'+','.join(str(values[k]) for k in backend.SPEED_KEYS)+'\n'
        def operation():
            if backend.link.exchange(packet) != 'OK': raise RuntimeError('ESP32 ไม่ยอมรับค่าความเร็ว')
            backend.speeds.update(values)
        self.run_background(operation, lambda _r:self.set_message('บันทึกความเร็วแล้ว'))
    def _angle_exchange(self, values=None):
        if values is not None:
            packet = 'SA'+','.join(str(values[k]) for k in backend.ANGLE_KEYS)+'\n'
            if backend.link.exchange(packet) != 'OK': raise RuntimeError('ESP32 ไม่ยอมรับค่ามุม')
        parts = backend.link.exchange('Q').split(',')
        if len(parts) != 5 or parts[0] != 'ANGLES': raise RuntimeError('ESP32 ตอบค่ามุมไม่ถูกต้อง')
        return dict(zip(backend.ANGLE_KEYS, map(int, parts[1:])))
    def load_angles(self):
        if backend.auto.state()['running']:
            self.set_message('หยุด Auto ก่อนอ่านค่ามุม', True); return
        def success(values):
            for k,v in values.items(): self.angle_vars[k].set(v)
            self.set_message('อ่านมุมแล้ว')
        self.run_background(self._angle_exchange, success)
    def save_angles(self):
        if backend.auto.state()['running']:
            self.set_message('หยุด Auto ก่อนปรับมุม', True); return
        try:
            values = {k:int(v.get()) for k,v in self.angle_vars.items()}
            if any(not 0 <= v <= 180 for v in values.values()): raise ValueError
        except (ValueError, tk.TclError): self.set_message('มุมต้องเป็นจำนวนเต็ม 0–180', True); return
        self.run_background(lambda:self._angle_exchange(values), lambda _r:self.set_message('บันทึกและตรวจสอบมุมแล้ว'))

    def _tick(self):
        if self.closing: return
        backend.touch_browser()
        with backend.field_frame_lock: main_jpeg, mask_jpeg = backend.field_frame, backend.mask_frame
        if main_jpeg is not None: self.image_canvas.show_jpeg(main_jpeg)
        elif not backend.field_running: self.image_canvas.clear_image()
        if backend.vision.mode == 'colour' and mask_jpeg is not None: self.mask_canvas.show_jpeg(mask_jpeg)
        state = backend.vision.state()
        self.field_progress.set(f"มุมสนาม {len(state['field']['arena'])}/4  •  วงสี {len(state['field']['holes'])}/6")
        auto = backend.auto.state(); host,port = backend.link.endpoint
        self.status_var.set(f"ESP32 {host}:{port} {backend.link.status}  •  กล้อง {'ON' if backend.field_running else 'OFF'}  •  {auto['message']}")
        if backend.field_error: self.message_var.set(backend.field_error)
        self.root.after(100, self._tick)
    def close(self):
        if self.closing: return
        self.closing = True; self.drive_stop(); backend.cleanup(); self.root.destroy()


def main():
    root = tk.Tk(); DesktopApp(root); root.mainloop()


if __name__ == '__main__': main()
