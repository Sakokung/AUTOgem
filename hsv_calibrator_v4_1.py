"""Live HSV picker + sliders for six gem colours.

Keys: 1-6 select colour, click gems to sample, U undo sample, C clear colour,
S save hsv_colour_config.json, Q/Esc quit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from auto_vision_gem_camera_v4_1 import open_camera
from auto_vision_gem_colors_v4_1 import (
    COLOURS, DISPLAY_COLOURS, load_arena_calibration, rectify_arena,
)


START = {
    'BLUE': (105, 14, 190, 65, 125, 105),
    'CYAN': (95, 16, 115, 90, 195, 70),
    'GREEN': (60, 18, 125, 105, 165, 100),
    'PURPLE': (145, 18, 145, 105, 130, 115),
    'ORANGE': (15, 13, 175, 80, 190, 70),
    'RED': (0, 13, 190, 70, 155, 105),
}
TRACKS = (
    ('H center', 179), ('H tolerance', 89),
    ('S center', 255), ('S tolerance', 255),
    ('V center', 255), ('V tolerance', 255),
    ('Min area', 2000), ('Max area', 20000), ('Morph kernel', 15),
)


def default_config():
    colours = {}
    for colour in COLOURS:
        hc, ht, sc, st, vc, vt = START[colour]
        colours[colour] = {
            'h_center': hc, 'h_tolerance': ht,
            's_center': sc, 's_tolerance': st,
            'v_center': vc, 'v_tolerance': vt,
            'min_area': 55, 'max_area': 4500, 'morph_kernel': 3,
        }
    return {'version': 1, 'colour_space': 'HSV',
            'reference_width': 1050, 'colours': colours}


def hue_distance(values, centre):
    difference = np.abs(values.astype(np.float32) - float(centre))
    return np.minimum(difference, 180.0 - difference)


def mask_for(hsv, item):
    h, s, v = cv2.split(hsv)
    mask = ((hue_distance(h, item['h_center']) <= item['h_tolerance'])
            & (np.abs(s.astype(np.int16) - item['s_center'])
               <= item['s_tolerance'])
            & (np.abs(v.astype(np.int16) - item['v_center'])
               <= item['v_tolerance'])).astype(np.uint8) * 255
    kernel = max(1, int(item['morph_kernel']))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((kernel, kernel), np.uint8))


class Calibrator:
    def __init__(self, output, patch_size=15):
        self.output = output
        self.patch_size = patch_size if patch_size % 2 else patch_size + 1
        self.config = default_config()
        self.selected = 0
        self.sample_batches = {colour: [] for colour in COLOURS}
        self.frame = None
        self.hsv = None
        self.display_scale = 1.0
        self.updating = False
        self.window = 'HSV Calibrator V4.1 | 1-6 colour | Click sample | S save'

    @property
    def colour(self):
        return COLOURS[self.selected]

    @property
    def item(self):
        return self.config['colours'][self.colour]

    def setup(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        for name, maximum in TRACKS:
            cv2.createTrackbar(name, self.window, 0, maximum, self.on_trackbar)
        self.sync_trackbars()

    def sync_trackbars(self):
        self.updating = True
        values = list(self.item.values())
        for (name, _maximum), value in zip(TRACKS, values):
            cv2.setTrackbarPos(name, self.window, int(value))
        self.updating = False

    def on_trackbar(self, _value):
        if self.updating:
            return
        keys = tuple(self.item)
        for (name, _maximum), key in zip(TRACKS, keys):
            self.item[key] = cv2.getTrackbarPos(name, self.window)
        self.item['h_tolerance'] = max(1, self.item['h_tolerance'])
        self.item['s_tolerance'] = max(1, self.item['s_tolerance'])
        self.item['v_tolerance'] = max(1, self.item['v_tolerance'])
        self.item['max_area'] = max(self.item['min_area'] + 1,
                                    self.item['max_area'])
        self.item['morph_kernel'] = max(1, self.item['morph_kernel'])

    def on_mouse(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        px, py = round(x / self.display_scale), round(y / self.display_scale)
        height, width = self.hsv.shape[:2]
        if not (0 <= px < width and 0 <= py < height):
            return
        radius = self.patch_size // 2
        patch = self.hsv[max(0, py-radius):min(height, py+radius+1),
                         max(0, px-radius):min(width, px+radius+1)].reshape(-1, 3)
        self.sample_batches[self.colour].append(patch.copy())
        self.recompute_from_samples()

    def recompute_from_samples(self):
        batches = self.sample_batches[self.colour]
        if not batches:
            return
        pixels = np.concatenate(batches, axis=0).astype(np.float32)
        angles = pixels[:, 0] * (2 * np.pi / 180.0)
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        h_center = int(round((mean_angle % (2*np.pi)) * 180.0 / (2*np.pi))) % 180
        s_center = int(round(np.median(pixels[:, 1])))
        v_center = int(round(np.median(pixels[:, 2])))
        h_tol = int(np.clip(np.percentile(hue_distance(pixels[:, 0], h_center), 97)+2, 3, 89))
        s_tol = int(np.clip(np.percentile(np.abs(pixels[:, 1]-s_center), 97)+10, 12, 255))
        v_tol = int(np.clip(np.percentile(np.abs(pixels[:, 2]-v_center), 97)+10, 12, 255))
        self.item.update(h_center=h_center, h_tolerance=h_tol,
                         s_center=s_center, s_tolerance=s_tol,
                         v_center=v_center, v_tolerance=v_tol)
        self.sync_trackbars()

    def select(self, index):
        self.selected = index
        self.sync_trackbars()

    def undo(self):
        batches = self.sample_batches[self.colour]
        if batches:
            batches.pop()
            if batches:
                self.recompute_from_samples()

    def clear(self):
        self.sample_batches[self.colour].clear()
        hc, ht, sc, st, vc, vt = START[self.colour]
        self.item.update(h_center=hc, h_tolerance=ht,
                         s_center=sc, s_tolerance=st,
                         v_center=vc, v_tolerance=vt)
        self.sync_trackbars()

    def save(self):
        self.config['reference_width'] = self.frame.shape[1]
        self.output.write_text(json.dumps(self.config, indent=2), encoding='utf-8')
        print('Saved HSV config:', self.output.resolve())

    def render(self, frame):
        self.frame = frame
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = mask_for(self.hsv, self.item)
        preview = frame.copy()
        accepted = 0
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.item['min_area'] <= area <= self.item['max_area']:
                x, y, width, height = cv2.boundingRect(contour)
                cv2.rectangle(preview, (x, y), (x+width, y+height),
                              DISPLAY_COLOURS[self.colour], 2)
                accepted += 1
        mh, mw = min(150, frame.shape[0]//3), min(260, frame.shape[1]//3)
        inset = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        preview[45:45+mh, frame.shape[1]-mw-8:frame.shape[1]-8] = inset
        overlay = preview.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.addWeighted(overlay, .60, preview, .40, 0, preview)
        samples = len(self.sample_batches[self.colour])
        text = (f'{self.selected+1} {self.colour} | clicks {samples} | '
                f'accepted blobs {accepted} | S save  U undo  C clear  Q quit')
        cv2.putText(preview, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (255, 255, 255), 2, cv2.LINE_AA)
        ratio = min(1.0, 1200/frame.shape[1], 650/frame.shape[0])
        self.display_scale = ratio
        if ratio < 1:
            preview = cv2.resize(preview, None, fx=ratio, fy=ratio,
                                 interpolation=cv2.INTER_AREA)
        return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=('auto', 'dshow', 'msmf'), default='dshow')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--arena-calibration', type=Path)
    parser.add_argument('--arena-width', type=int, default=1050)
    parser.add_argument('--arena-height', type=int, default=600)
    parser.add_argument('--output', type=Path, default=Path('hsv_colour_config.json'))
    parser.add_argument('--patch-size', type=int, default=15)
    args = parser.parse_args()
    arena = None
    if args.arena_calibration:
        try:
            arena = load_arena_calibration(args.arena_calibration)
        except ValueError as exc:
            parser.error(str(exc))
    camera = open_camera(args.camera, args.backend, args.width, args.height)
    if camera is None:
        parser.error(f'Cannot open camera {args.camera}')
    calibrator = Calibrator(args.output, args.patch_size)
    calibrator.setup()
    print('1-6 select colour; click several real gems; sliders update live; S saves.')
    try:
        while True:
            ok, raw = camera.read()
            if not ok or raw is None:
                print('Camera frame could not be read.')
                break
            frame = (rectify_arena(raw, arena, args.arena_width, args.arena_height)
                     if arena is not None else raw)
            cv2.imshow(calibrator.window, calibrator.render(frame))
            key = cv2.waitKey(1) & 0xFF
            if ord('1') <= key <= ord('6'):
                calibrator.select(key - ord('1'))
            elif key in (ord('u'), ord('U')):
                calibrator.undo()
            elif key in (ord('c'), ord('C')):
                calibrator.clear()
            elif key in (ord('s'), ord('S')):
                calibrator.save()
            elif key in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(calibrator.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print('Camera released.')


if __name__ == '__main__':
    main()
