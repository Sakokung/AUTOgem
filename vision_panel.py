"""Shared full-frame vision state for the Flask control panel.

All image clicks and field geometry are normalized to the original camera frame.
There is deliberately no arena warp, crop, or motor automation here.
"""
import copy
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from auto_vision_gem_colors_v4_1 import (
    COLOURS, DISPLAY_COLOURS, configured_hsv_masks, detect,
)
from auto_vision_gem_camera_v4_1 import detect_aruco, choose_nearest_target
from hsv_calibrator_v4_1 import default_config, hue_distance


KEYS = ('h_center', 'h_tolerance', 's_center', 's_tolerance',
        'v_center', 'v_tolerance', 'min_area', 'max_area', 'morph_kernel')
LIMITS = (179, 89, 255, 255, 255, 255, 2000, 20000, 15)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def valid_config(config):
    if not isinstance(config, dict) or config.get('version') != 1 or config.get('colour_space') != 'HSV':
        raise ValueError('ไฟล์สีต้องเป็น HSV version 1')
    if set(config.get('colours', {})) != set(COLOURS):
        raise ValueError('ไฟล์สีต้องมีทั้ง 6 สี')
    for colour in COLOURS:
        values = config['colours'][colour]
        if set(values) != set(KEYS):
            raise ValueError(f'ค่า {colour} ไม่ครบ')
        for key, limit in zip(KEYS, LIMITS):
            value = values[key]
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f'{colour}: {key} ไม่ถูกต้อง')
        if (values['h_tolerance'] < 1 or values['s_tolerance'] < 1 or
                values['v_tolerance'] < 1 or values['morph_kernel'] < 1 or
                values['max_area'] <= values['min_area']):
            raise ValueError(f'ช่วงสีหรือพื้นที่ของ {colour} ไม่ถูกต้อง')
    if type(config.get('reference_width')) is not int or config['reference_width'] < 2:
        raise ValueError('reference_width ไม่ถูกต้อง')
    return config


def valid_point(point):
    if (not isinstance(point, (list, tuple)) or len(point) != 2 or
            any(type(value) not in (int, float) or not np.isfinite(value) or
                not 0 <= value <= 1 for value in point)):
        raise ValueError('พิกัดคลิกไม่ถูกต้อง')
    return [float(point[0]), float(point[1])]


def pixel(point, shape):
    h, w = shape[:2]
    return (round(point[0] * (w - 1)), round(point[1] * (h - 1)))


class VisionPanel:
    def __init__(self, config_path, field_path):
        self.lock = threading.RLock()
        self.config_path = Path(config_path)
        self.field_path = Path(field_path)
        self.config = default_config()
        self.saved_config = copy.deepcopy(self.config)
        self.field = {'version': 1, 'arena': [], 'holes': {}}
        self.mode = 'view'
        self.colour = 'RED'
        self.hole_colour = 'RED'
        self.hole_pending = None
        self.samples = {colour: [] for colour in COLOURS}
        self.frame = None
        self.frame_captured_at = 0.0
        self.dirty = False
        self.frame_index = 0
        self.last_preview = None
        self.observation = None
        self.delivery_colour = None
        if self.config_path.exists():
            self.config = valid_config(json.loads(self.config_path.read_text(encoding='utf-8-sig')))
            self.saved_config = copy.deepcopy(self.config)
        if self.field_path.exists():
            field = json.loads(self.field_path.read_text(encoding='utf-8-sig'))
            if field.get('version') != 1:
                raise ValueError('field_config.json version ไม่ถูกต้อง')
            arena = [valid_point(p) for p in field.get('arena', [])]
            if len(arena) not in (0, 4):
                raise ValueError('กรอบสนามต้องมี 4 มุม')
            holes = field.get('holes', {})
            if not isinstance(holes, dict) or any(c not in COLOURS for c in holes):
                raise ValueError('ชื่อสีวงไม่ถูกต้อง')
            self.field = {'version': 1, 'arena': arena, 'holes': {}}
            for c, item in holes.items():
                self.field['holes'][c] = {'center': valid_point(item['center']),
                                          'edge': valid_point(item['edge'])}

    def state(self):
        with self.lock:
            return {'mode': self.mode, 'colour': self.colour,
                    'hole_colour': self.hole_colour, 'hole_pending': self.hole_pending,
                    'values': copy.deepcopy(self.config['colours'][self.colour]),
                    'samples': {c: len(v) for c, v in self.samples.items()},
                    'field': copy.deepcopy(self.field), 'dirty': self.dirty,
                    'frame_ready': self.frame is not None}

    def set_mode(self, mode):
        if mode not in ('view', 'colour', 'arena', 'hole'):
            raise ValueError('โหมดไม่ถูกต้อง')
        with self.lock:
            self.mode = mode
            self.hole_pending = None
            if mode != 'view':
                self.observation = None

    def auto_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.observation)

    def set_delivery_guide(self, colour):
        if colour is not None and colour not in COLOURS:
            raise ValueError('Invalid delivery guide colour')
        with self.lock:
            self.delivery_colour = colour

    def select(self, colour, hole=False):
        if colour not in COLOURS:
            raise ValueError('สีไม่ถูกต้อง')
        with self.lock:
            if hole:
                self.hole_colour = colour
                self.hole_pending = None
            else:
                self.colour = colour

    def sliders(self, values):
        if not isinstance(values, dict) or set(values) != set(KEYS):
            raise ValueError('ต้องส่ง Slider ทั้ง 9 ค่า')
        with self.lock:
            candidate = copy.deepcopy(self.config)
            candidate['colours'][self.colour] = dict(values)
            valid_config(candidate)
            self.config = candidate
            self.dirty = True

    def undo(self):
        with self.lock:
            if self.samples[self.colour]:
                self.samples[self.colour].pop()
                if self.samples[self.colour]:
                    self._recompute(self.colour)
                else:
                    self.config['colours'][self.colour] = copy.deepcopy(
                        self.saved_config['colours'][self.colour])
                self.dirty = True

    def reset_colour(self):
        with self.lock:
            self.samples[self.colour].clear()
            self.config['colours'][self.colour] = copy.deepcopy(
                default_config()['colours'][self.colour])
            self.dirty = True

    def save_colours(self):
        with self.lock:
            if self.frame is None:
                raise ValueError('เปิดกล้องสนามก่อนบันทึกสี')
            self.config['reference_width'] = self.frame.shape[1]
            valid_config(self.config)
            atomic_json(self.config_path, self.config)
            self.saved_config = copy.deepcopy(self.config)
            self.dirty = False

    def save_field(self):
        with self.lock:
            if len(self.field['arena']) != 4 or len(self.field['holes']) != 6:
                raise ValueError('คลิก 4 มุมและวงทั้ง 6 สีให้ครบก่อนบันทึก')
            arena = np.array(self.field['arena'], np.float32)
            if not cv2.isContourConvex(arena) or cv2.contourArea(arena) < .05:
                raise ValueError('มุมสนามต้องเรียงรอบขอบ ไม่ไขว้กัน และครอบพื้นที่จริง')
            for colour, hole in self.field['holes'].items():
                if cv2.pointPolygonTest(arena, tuple(hole['center']), False) < 0:
                    raise ValueError(f'กลางวง {colour} อยู่นอกกรอบสนาม')
            atomic_json(self.field_path, self.field)

    def reset_arena(self):
        with self.lock:
            self.field['arena'] = []

    def undo_arena(self):
        with self.lock:
            if self.field['arena']:
                self.field['arena'].pop()

    def clear_hole(self):
        with self.lock:
            self.field['holes'].pop(self.hole_colour, None)
            self.hole_pending = None

    def set_frame(self, frame, captured_at=None):
        with self.lock:
            self.frame = frame.copy()
            self.frame_captured_at = (time.monotonic() if captured_at is None
                                      else float(captured_at))
            self.frame_index += 1

    def click(self, point):
        point = valid_point(point)
        with self.lock:
            if self.frame is None:
                raise ValueError('เปิดกล้องสนามก่อนคลิก')
            if self.mode == 'arena':
                if len(self.field['arena']) == 4:
                    raise ValueError('ครบ 4 มุมแล้ว กดเริ่มใหม่ถ้าต้องการแก้')
                self.field['arena'].append(point)
            elif self.mode == 'hole':
                if self.hole_pending is None:
                    self.hole_pending = point
                else:
                    cx, cy = pixel(self.hole_pending, self.frame.shape)
                    ex, ey = pixel(point, self.frame.shape)
                    if np.hypot(ex - cx, ey - cy) < 5:
                        raise ValueError('คลิกขอบวงให้ห่างจากจุดกลางอย่างน้อย 5 px')
                    self.field['holes'][self.hole_colour] = {
                        'center': self.hole_pending, 'edge': point}
                    self.hole_pending = None
            elif self.mode == 'colour':
                self._sample(point)
            else:
                raise ValueError('เลือกแท็บปรับสีหรือตั้งสนามก่อนคลิก')

    def _sample(self, point):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        x, y = pixel(point, hsv.shape)
        h, w = hsv.shape[:2]
        # The 5x5 click core identifies the gem. Reject differently coloured
        # floor pixels from the surrounding 15x15 sample patch.
        core = hsv[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
        patch = hsv[max(0, y-7):min(h, y+8), max(0, x-7):min(w, x+8)]
        angles = core[:, :, 0].astype(np.float32) * (2*np.pi/180)
        seed_h = round(np.arctan2(np.sin(angles).mean(),
                                 np.cos(angles).mean()) % (2*np.pi) * 180/(2*np.pi)) % 180
        seed_s, seed_v = np.median(core[:, :, 1:], axis=(0, 1))
        if seed_s < 35:
            raise ValueError('จุดที่คลิกสีจางเกินไป กรุณาคลิกกลางหิน')
        keep = ((hue_distance(patch[:, :, 0], seed_h) <= 15) &
                (np.abs(patch[:, :, 1].astype(float)-seed_s) <= 65) &
                (np.abs(patch[:, :, 2].astype(float)-seed_v) <= 65) &
                (patch[:, :, 1] >= 35))
        chosen = patch[keep] if np.count_nonzero(keep) >= 12 else core.reshape(-1, 3)
        self.samples[self.colour].append(chosen.copy())
        self._recompute(self.colour)
        self.dirty = True

    def _recompute(self, colour):
        pixels = np.concatenate(self.samples[colour], axis=0).astype(np.float32)
        angles = pixels[:, 0] * (2*np.pi/180)
        centre_h = round(np.arctan2(np.sin(angles).mean(),
                                   np.cos(angles).mean()) % (2*np.pi) * 180/(2*np.pi)) % 180
        centre_s = round(float(np.median(pixels[:, 1])))
        centre_v = round(float(np.median(pixels[:, 2])))
        values = self.config['colours'][colour]
        values.update(h_center=centre_h,
                      h_tolerance=int(np.clip(np.percentile(hue_distance(pixels[:, 0], centre_h), 95)+3, 3, 89)),
                      s_center=centre_s,
                      s_tolerance=int(np.clip(np.percentile(np.abs(pixels[:, 1]-centre_s), 95)+10, 12, 255)),
                      v_center=centre_v,
                      v_tolerance=int(np.clip(np.percentile(np.abs(pixels[:, 2]-centre_v), 95)+10, 12, 255)))

    def render(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
            shown = frame.copy()
            h, w = frame.shape[:2]
            active_config = self.config if self.mode == 'colour' else self.saved_config
            if self.mode == 'colour':
                self.observation = None
                mask = configured_hsv_masks(frame, active_config)[self.colour]
                tint = np.zeros_like(frame)
                tint[:] = DISPLAY_COLOURS[self.colour]
                shown[mask > 0] = cv2.addWeighted(frame, .55, tint, .45, 0)[mask > 0]
            else:
                # Same HSV detector used in the later run mode; this is a
                # visual check only and never sends motion commands.
                markers, car_mask = detect_aruco(frame)
                _, gems, _, _ = detect(frame, hsv_config=active_config,
                                       extra_excluded=car_mask)
                polygon = (np.array([pixel(p, frame.shape) for p in self.field['arena']],
                                    np.int32) if len(self.field['arena']) == 4 else None)
                available_gems = []
                for gem in gems:
                    x, y = gem['x'], gem['y']
                    if polygon is not None and cv2.pointPolygonTest(polygon, (x, y), False) < 0:
                        continue
                    if self._inside_hole((x, y), frame.shape):
                        continue
                    available_gems.append(gem)
                    c = DISPLAY_COLOURS[gem['colour']]
                    cv2.circle(shown, (x, y), 7, c, 2)
                    cv2.putText(shown, gem['colour'][0], (x+8, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, c, 2, cv2.LINE_AA)
                matching_robots = [marker for marker in markers
                                   if marker['aruco_id'] == 0]
                robot = matching_robots[0] if len(matching_robots) == 1 else None
                gripper_gems = []
                if robot is not None:
                    capture = robot['gripper_capture_polygon']
                    gripper_gems = [gem for gem in available_gems
                                    if cv2.pointPolygonTest(
                                        capture, (gem['x'], gem['y']), False) >= 0]
                holes = {}
                for colour, hole in self.field['holes'].items():
                    centre = pixel(hole['center'], frame.shape)
                    edge = pixel(hole['edge'], frame.shape)
                    holes[colour] = {
                        'x': centre[0], 'y': centre[1],
                        'radius': float(np.hypot(edge[0]-centre[0],
                                                 edge[1]-centre[1])),
                    }
                for marker in markers:
                    cv2.polylines(shown, [marker['corners'].reshape(-1, 1, 2)],
                                  True, (0, 255, 0), 2)
                    cv2.arrowedLine(shown, (marker['x'], marker['y']),
                                    (marker['gripper_tip_x'], marker['gripper_tip_y']),
                                    (0, 230, 255), 2, tipLength=.15)
                    capture_colour = ((0, 255, 0) if gripper_gems and marker is robot
                                      else (0, 165, 255))
                    cv2.polylines(shown, [marker['gripper_capture_polygon']], True,
                                  capture_colour, 2, cv2.LINE_AA)
                    cv2.putText(shown, f"ArUco {marker['aruco_id']}",
                                (marker['x']+8, marker['y']-12),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
                if (robot is not None and self.delivery_colour in holes):
                    hole = holes[self.delivery_colour]
                    start = (robot['gripper_tip_x'], robot['gripper_tip_y'])
                    destination = (hole['x'], hole['y'])
                    colour = DISPLAY_COLOURS[self.delivery_colour]
                    cv2.arrowedLine(shown, start, destination, colour, 5,
                                    cv2.LINE_AA, tipLength=.08)
                    label = f"DELIVER {self.delivery_colour}"
                    cv2.putText(shown, label,
                                (min(start[0], destination[0])+8,
                                 max(24, min(start[1], destination[1])-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, colour, 2,
                                cv2.LINE_AA)
                selected = (None if self.delivery_colour is not None else
                            choose_nearest_target(markers, available_gems,
                                                  robot_id=0))
                self.observation = {
                    'time': time.monotonic(), 'frame_index': self.frame_index,
                    'captured_at': self.frame_captured_at,
                    # Keep the robot pose available even when no gem is visible.
                    # Auto mode can then continue toward its last locked target.
                    'robot': copy.deepcopy(robot),
                    'gems': [dict(gem) for gem in available_gems],
                    'gripper_gems': [dict(gem) for gem in gripper_gems],
                    'holes': holes,
                    'target': dict(selected[1]) if selected else None,
                }
                if selected is not None:
                    robot, gem, distance_px = selected
                    tip = (robot['gripper_tip_x'], robot['gripper_tip_y'])
                    point = (gem['x'], gem['y'])
                    cv2.line(shown, tip, point, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(shown, point, 17, (255, 255, 255), 3, cv2.LINE_AA)
                    label = f"TARGET: {gem['colour']}  {distance_px:.0f} px"
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                 .60, 2)[0][0]
                    cv2.rectangle(shown, (8, 8), (min(w-1, text_width+28), 47),
                                  (15, 15, 15), -1)
                    cv2.putText(shown, label, (16, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                .60, (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.field['arena']) >= 2:
                pts = np.array([pixel(p, frame.shape) for p in self.field['arena']], np.int32)
                cv2.polylines(shown, [pts], len(pts) == 4, (0, 240, 240), 3)
            for i, point in enumerate(self.field['arena'], 1):
                q = pixel(point, frame.shape)
                cv2.circle(shown, q, 6, (0, 255, 255), -1)
                cv2.putText(shown, str(i), (q[0]+8, q[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
            for colour, hole in self.field['holes'].items():
                c = pixel(hole['center'], frame.shape)
                e = pixel(hole['edge'], frame.shape)
                radius = round(np.hypot(e[0]-c[0], e[1]-c[1]))
                cv2.circle(shown, c, radius, DISPLAY_COLOURS[colour], 3)
                cv2.drawMarker(shown, c, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(shown, colour, (c[0]-30, max(20, c[1]-radius-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
            if self.hole_pending is not None:
                cv2.drawMarker(shown, pixel(self.hole_pending, frame.shape),
                               (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
            return shown

    def colour_mask(self):
        """Full-size binary preview of the selected HSV colour."""
        with self.lock:
            if self.mode != 'colour' or self.frame is None:
                return None
            return configured_hsv_masks(self.frame, self.config)[self.colour]

    def _inside_hole(self, point, shape):
        for hole in self.field['holes'].values():
            c, e = pixel(hole['center'], shape), pixel(hole['edge'], shape)
            if np.hypot(point[0]-c[0], point[1]-c[1]) <= np.hypot(e[0]-c[0], e[1]-c[1]):
                return True
        return False
