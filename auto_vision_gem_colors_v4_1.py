"""Gem Colours V4.1: arena + persistent hole exclusion + colour detection.

Requires: pip install opencv-python numpy
Run with the trained model:
python auto_vision_gem_colors_v4_1.py --image "f1.png" --model gem_color_model.npz
Omit --model to use the original fixed-HSV fallback.
First arena setup: add --calibrate-arena and click four inner corners.
Later images: add --arena-calibration arena_calibration.npz.
First hole setup (after arena rectification): add --calibrate-holes and click
the centre and edge of each of the six holes in the prompted colour order.
Later images: add --hole-calibration hole_calibration.npz. A gem whose centre
is inside a saved hole (plus the safety margin) is ignored and not counted.
Use --no-gui for batch processing. Outputs an overview, six colour panels,
a zoom, per-gem CSV, and per-colour counts CSV. Keys 1/2/3 switch views;
Q/Esc or closing the window exits.
Use --select-roi to draw a rectangle around a pile (ENTER accepts).
Use --roi X Y WIDTH HEIGHT for headless pile selection.
Counts describe detected visible centres in the selected area, not hidden gems.
Coordinates in cm are approximate: assumes the entire input is a rectified
210 x 120 cm arena, including the Start Zone. This is NOT camera calibration.
BLUE means dark blue; CYAN means light blue. Thresholds are reference-image
settings and need validation under the real camera/lighting.
The supplied reference test yields 54 candidates; no fixed count is enforced.
Brightness-assisted splits can be affected by shadows on a single object.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = ("BLUE", "CYAN", "GREEN", "PURPLE", "ORANGE", "RED")
REFERENCE_WIDTH = 741.0
PREFIX = dict(zip(COLOURS, ('B', 'C', 'G', 'P', 'O', 'R')))
DISPLAY_COLOURS = {'BLUE': (170, 65, 15), 'CYAN': (230, 185, 65),
                  'GREEN': (70, 160, 65), 'PURPLE': (145, 60, 145),
                  'ORANGE': (40, 145, 230), 'RED': (55, 55, 205)}


def order_corners(points):
    """Return four points in TL, TR, BR, BL order."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError('Exactly four arena corners are required')
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).reshape(-1)
    ordered = np.array([
        points[np.argmin(total)],
        points[np.argmin(difference)],
        points[np.argmax(total)],
        points[np.argmax(difference)],
    ], dtype=np.float32)
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError('Arena corners must be four different points')
    return ordered


def select_arena_corners(frame):
    """Interactively collect the four inner field corners in any order."""
    fh, fw = frame.shape[:2]
    ratio = min(1.0, 1400 / fw, 820 / fh)
    preview = cv2.resize(frame, (round(fw * ratio), round(fh * ratio)),
                         interpolation=cv2.INTER_AREA)
    points = []
    window = 'Arena calibration - click 4 inner corners'

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / ratio, y / ratio))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print('Click the four INNER arena corners in any order.')
    print('U: reset points | Enter: accept four points | Q/Esc: cancel')
    while True:
        shown = preview.copy()
        for number, (x, y) in enumerate(points, 1):
            px, py = round(x * ratio), round(y * ratio)
            cv2.circle(shown, (px, py), 8, (0, 255, 255), -1)
            cv2.putText(shown, str(number), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(shown, str(number), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(shown, f'Points {len(points)}/4', (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(shown, f'Points {len(points)}/4', (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(window, shown)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('u'), ord('U')):
            points.clear()
        elif key in (13, 10) and len(points) == 4:
            break
        elif key in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow(window)
            return None
    cv2.destroyWindow(window)
    return order_corners(points)


def save_arena_calibration(path, corners, frame_shape):
    """Save normalized corners so calibration scales with input resolution."""
    height, width = frame_shape[:2]
    normalized = corners / np.array([width, height], dtype=np.float32)
    np.savez_compressed(path, corners_normalized=normalized,
                        arena_cm=np.array([210.0, 120.0]))
    print('Saved arena calibration:', path.resolve())


def load_arena_calibration(path):
    try:
        data = np.load(path)
        normalized = np.asarray(data['corners_normalized'], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'Cannot load arena calibration {path}: {exc}') from exc
    if normalized.shape != (4, 2):
        raise ValueError('Arena calibration must contain four normalized corners')
    return normalized


def rectify_arena(frame, normalized_corners, output_width=1050,
                  output_height=600):
    """Warp the inner 210x120 cm arena to a top-down rectangle."""
    height, width = frame.shape[:2]
    source = normalized_corners * np.array([width, height], dtype=np.float32)
    source = order_corners(source)
    destination = np.array([
        [0, 0], [output_width - 1, 0],
        [output_width - 1, output_height - 1], [0, output_height - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, matrix, (output_width, output_height))


def select_holes(frame):
    """Collect centre + edge clicks for six holes in COLOURS order."""
    fh, fw = frame.shape[:2]
    ratio = min(1.0, 1400 / fw, 820 / fh)
    preview = cv2.resize(frame, (round(fw * ratio), round(fh * ratio)),
                         interpolation=cv2.INTER_AREA)
    clicks = []
    cursor = [0, 0]
    window = 'Hole calibration - centre then edge for each colour'

    def on_mouse(event, x, y, _flags, _userdata):
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 12:
            clicks.append((x / ratio, y / ratio))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print('For each prompted colour: click hole CENTRE, then click its EDGE.')
    print('U: undo | R: reset | Enter: save after all 12 clicks | Q/Esc: cancel')
    while True:
        shown = preview.copy()
        complete = len(clicks) // 2
        for index in range(complete):
            cx, cy = clicks[index * 2]
            ex, ey = clicks[index * 2 + 1]
            centre = (round(cx * ratio), round(cy * ratio))
            radius = max(1, round(np.hypot(ex - cx, ey - cy) * ratio))
            cv2.circle(shown, centre, radius, DISPLAY_COLOURS[COLOURS[index]], 2)
            cv2.drawMarker(shown, centre, (255, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)
            cv2.putText(shown, COLOURS[index],
                        (max(0, centre[0] - 35), max(20, centre[1] - radius - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 3,
                        cv2.LINE_AA)
            cv2.putText(shown, COLOURS[index],
                        (max(0, centre[0] - 35), max(20, centre[1] - radius - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if len(clicks) < 12:
            colour = COLOURS[len(clicks) // 2]
            stage = 'CENTRE' if len(clicks) % 2 == 0 else 'EDGE'
            prompt = f'{colour}: click {stage}  ({len(clicks)}/12)'
            if len(clicks) % 2 == 1:
                cx, cy = clicks[-1]
                centre = (round(cx * ratio), round(cy * ratio))
                radius = max(1, round(np.hypot(cursor[0] - centre[0],
                                              cursor[1] - centre[1])))
                cv2.circle(shown, centre, radius,
                           DISPLAY_COLOURS[colour], 1)
        else:
            prompt = 'All holes ready - press Enter to save'
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(shown, prompt, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    .70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, shown)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('u'), ord('U')) and clicks:
            clicks.pop()
        elif key in (ord('r'), ord('R')):
            clicks.clear()
        elif key in (13, 10) and len(clicks) == 12:
            break
        elif key in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow(window)
            return None
    cv2.destroyWindow(window)
    px_per_cm_x = (fw - 1) / 210.0
    px_per_cm_y = (fh - 1) / 120.0
    centres_cm, radii_cm = [], []
    for index in range(6):
        cx, cy = clicks[index * 2]
        ex, ey = clicks[index * 2 + 1]
        centres_cm.append((cx / px_per_cm_x, cy / px_per_cm_y))
        radii_cm.append(np.hypot((ex - cx) / px_per_cm_x,
                                 (ey - cy) / px_per_cm_y))
    return (np.asarray(centres_cm, dtype=np.float32),
            np.asarray(radii_cm, dtype=np.float32))


def save_hole_calibration(path, centres_cm, radii_cm):
    np.savez_compressed(path, labels=np.asarray(COLOURS),
                        centres_cm=np.asarray(centres_cm, dtype=np.float32),
                        radii_cm=np.asarray(radii_cm, dtype=np.float32),
                        arena_cm=np.array([210.0, 120.0], dtype=np.float32))
    print('Saved hole calibration:', path.resolve())


def load_hole_calibration(path):
    try:
        data = np.load(path)
        labels = tuple(str(item) for item in data['labels'])
        centres = np.asarray(data['centres_cm'], dtype=np.float32)
        radii = np.asarray(data['radii_cm'], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'Cannot load hole calibration {path}: {exc}') from exc
    if labels != COLOURS or centres.shape != (6, 2) or radii.shape != (6,):
        raise ValueError('Hole calibration must contain all six labelled holes')
    if not np.isfinite(centres).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Hole calibration contains invalid coordinates or radii')
    return centres, radii


def make_hole_mask(shape, centres_cm, radii_cm, margin_cm):
    """Create six elliptical masks in a rectified 210 x 120 cm frame."""
    height, width = shape[:2]
    sx, sy = (width - 1) / 210.0, (height - 1) / 120.0
    mask = np.zeros((height, width), np.uint8)
    for (x_cm, y_cm), radius_cm in zip(centres_cm, radii_cm):
        centre = (round(x_cm * sx), round(y_cm * sy))
        radius = max(0.1, float(radius_cm) + margin_cm)
        axes = (max(1, round(radius * sx)), max(1, round(radius * sy)))
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
    return mask


def draw_hole_zones(image, centres_cm, radii_cm, margin_cm):
    """Draw saved no-count zones without obscuring the camera image."""
    height, width = image.shape[:2]
    sx, sy = (width - 1) / 210.0, (height - 1) / 120.0
    for colour, (x_cm, y_cm), radius_cm in zip(COLOURS, centres_cm, radii_cm):
        centre = (round(x_cm * sx), round(y_cm * sy))
        radius = max(0.1, float(radius_cm) + margin_cm)
        axes = (max(1, round(radius * sx)), max(1, round(radius * sy)))
        cv2.ellipse(image, centre, axes, 0, 0, 360,
                    DISPLAY_COLOURS[colour], 2, cv2.LINE_AA)
        cv2.drawMarker(image, centre, (235, 235, 235),
                       cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(image, f'{PREFIX[colour]} HOLE',
                    (max(0, centre[0] - 30), max(18, centre[1] - axes[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f'{PREFIX[colour]} HOLE',
                    (max(0, centre[0] - 30), max(18, centre[1] - axes[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (20, 20, 20), 1,
                    cv2.LINE_AA)


def load_colour_model(path):
    """Load and validate the LAB Gaussian colour model."""
    try:
        model = np.load(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot load colour model {path}: {exc}') from exc
    required = {'labels', 'means', 'inv_covs', 'limits', 'colour_space'}
    missing = required.difference(model.files)
    if missing:
        raise ValueError(f'Colour model is missing: {sorted(missing)}')
    labels = tuple(str(item) for item in model['labels'])
    if labels != COLOURS:
        raise ValueError(f'Expected labels {COLOURS}, found {labels}')
    if str(model['colour_space']) != 'LAB':
        raise ValueError('Only a LAB colour model is supported')
    means = np.asarray(model['means'], dtype=np.float32)
    inv_covs = np.asarray(model['inv_covs'], dtype=np.float32)
    limits = np.asarray(model['limits'], dtype=np.float32)
    if (means.shape != (6, 3) or inv_covs.shape != (6, 3, 3)
            or limits.shape != (6,)):
        raise ValueError('Invalid colour-model array shapes')
    return {'labels': labels, 'means': means,
            'inv_covs': inv_covs, 'limits': limits}


def trained_colour_masks(frame, model):
    """Classify pixels with the trained LAB Mahalanobis model."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    distances = []
    for mean, inverse in zip(model['means'], model['inv_covs']):
        delta = lab - mean
        distances.append(np.einsum('...i,ij,...j->...',
                                   delta, inverse, delta, optimize=True))
    distances = np.stack(distances, axis=-1)
    best = np.argmin(distances, axis=-1)
    best_distance = np.take_along_axis(
        distances, best[..., None], axis=-1)[..., 0]
    accepted = best_distance <= model['limits'][best]
    return {
        colour: ((best == index) & accepted).astype(np.uint8) * 255
        for index, colour in enumerate(COLOURS)
    }


def load_hsv_config(path):
    """Load HSV centres/tolerances produced by hsv_calibrator_v4_1.py."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot load HSV config {path}: {exc}') from exc
    if data.get('colour_space') != 'HSV' or data.get('version') != 1:
        raise ValueError('HSV config must have version 1 and colour_space HSV')
    colours = data.get('colours', {})
    if tuple(colours) != COLOURS:
        raise ValueError(f'HSV config must contain colours in order: {COLOURS}')
    required = ('h_center', 'h_tolerance', 's_center', 's_tolerance',
                'v_center', 'v_tolerance', 'min_area', 'max_area',
                'morph_kernel')
    for colour in COLOURS:
        if any(key not in colours[colour] for key in required):
            raise ValueError(f'HSV config for {colour} is incomplete')
    return data


def configured_hsv_masks(frame, config):
    """Create exclusive HSV masks; overlaps go to the nearest HSV centre."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    # Keep only the current winner instead of stacking six full-frame boolean
    # and float arrays.  Besides using much less memory, this matters in the
    # live view where this function runs for every camera frame.
    best = np.full(h.shape, -1, np.int8)
    best_score = np.full(h.shape, np.inf, np.float32)
    for index, colour in enumerate(COLOURS):
        item = config['colours'][colour]
        dh = np.minimum(np.abs(h - item['h_center']),
                        180.0 - np.abs(h - item['h_center']))
        ds = np.abs(s - item['s_center'])
        dv = np.abs(v - item['v_center'])
        ht = max(float(item['h_tolerance']), 1.0)
        st = max(float(item['s_tolerance']), 1.0)
        vt = max(float(item['v_tolerance']), 1.0)
        accepted = (dh <= ht) & (ds <= st) & (dv <= vt)
        score = (dh / ht) ** 2 + (ds / st) ** 2 + (dv / vt) ** 2
        # Strict comparison preserves np.argmin's first-colour tie breaking.
        take = accepted & (score < best_score)
        best[take] = index
        best_score[take] = score[take]
    return {colour: (best == index).astype(np.uint8) * 255
            for index, colour in enumerate(COLOURS)}


def colour_masks(hsv, blue_sat=165):
    """One pixel has at most one class; red wraps around the hue scale."""
    h, s, v = cv2.split(hsv)
    blue_family = (h >= 86) & (h < 125) & (s >= 65) & (v >= 45)
    tests = [
        blue_family & (s >= blue_sat),
        blue_family & (s < blue_sat) & (v >= 100),
        (h >= 35) & (h < 86) & (s >= 55) & (v >= 50),
        (h >= 125) & (h < 170) & (s >= 40) & (v >= 40),
        (h >= 10) & (h <= 25) & (s >= 60) & (v >= 80),
        ((h < 10) | (h >= 170)) & (s >= 70) & (v >= 60),
    ]
    return {c: t.astype(np.uint8) * 255 for c, t in zip(COLOURS, tests)}


def contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def detect_targets(masks, scale):
    shape = next(iter(masks.values())).shape
    excluded = np.zeros(shape, np.uint8)
    targets = {}
    occupied = np.zeros(shape, np.uint8)
    for colour, mask in masks.items():
        candidates = []
        for contour in contours(mask):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
            if 1800 * scale**2 <= area <= 7000 * scale**2 and circularity >= .60:
                filled = np.zeros(shape, np.uint8)
                cv2.drawContours(filled, [contour], -1, 255, -1)
                interior = filled > 0
                size = np.count_nonzero(interior)
                # A thin cyan rim around a blue disk is not another target.
                if np.count_nonzero(mask[interior]) / size < .55:
                    continue
                if np.count_nonzero(occupied[interior]) / size > .30:
                    continue
                candidates.append((circularity, area, contour))
        if not candidates:
            continue
        contour = max(candidates, key=lambda item: item[:2])[2]
        m = cv2.moments(contour)
        center = (round(m['m10'] / m['m00']), round(m['m01'] / m['m00']))
        targets[colour] = {'center': center, 'contour': contour}
        cv2.drawContours(excluded, [contour], -1, 255, -1)
        cv2.drawContours(occupied, [contour], -1, 255, -1)
    margin = max(1, round(7 * scale))
    excluded = cv2.dilate(excluded, np.ones((2 * margin + 1,) * 2, np.uint8))
    return targets, excluded


def blob_centers(component, min_distance, peak_relative, require_neck=False,
                 brightness=None):
    """Padded distance map, plateau grouping, then strongest-first NMS.

    Padding supplies a real background at every crop edge. Peak height ranks
    candidates; component scan order never decides which peak to retain.
    """
    padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    # Fallback appearance evidence: a dark contact seam can separate two
    # visible faces even when their binary silhouettes have no clear neck.
    if brightness is not None:
        values = brightness.astype(np.float32)
        peak_value = max(float(values[component > 0].max()), 1.0)
        dist = dist * (values / peak_value)
    radius = max(1, round(min_distance / (9 if brightness is not None else 3)))
    local_max = cv2.dilate(dist, np.ones((2 * radius + 1,) * 2, np.uint8))
    peak_mask = ((dist >= local_max - 1e-6) &
                 (dist >= peak_relative * dist.max()) & (dist > 0)).astype(np.uint8)
    n, labels = cv2.connectedComponents(peak_mask)
    peaks = []
    for label in range(1, n):
        yy, xx = np.where(labels == label)
        # Choose an actual foreground peak nearest the plateau's centre.
        strengths = dist[yy, xx]
        best = np.flatnonzero(strengths >= strengths.max() - 1e-6)
        idx = best[np.argmin((xx[best]-xx.mean())**2 + (yy[best]-yy.mean())**2)]
        peaks.append((float(strengths[idx]), int(xx[idx]), int(yy[idx])))
    kept = []
    for strength, x, y in sorted(peaks, reverse=True):
        if all((x-kx)**2 + (y-ky)**2 >= min_distance**2 for _, kx, ky in kept):
            kept.append((strength, x, y))
    if not kept:
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        kept = [(float(dist[y, x]), int(x), int(y))]
    if not require_neck:
        return kept
    # A second peak needs a real neck: the two cores must disconnect
    # above 75% of the weaker peak height. A shallow ripple in a single
    # elongated gem therefore does not automatically create another gem.
    separated = []
    for strength, x, y in kept:
        level = (.80 if brightness is not None else .75) * strength
        _, core_labels = cv2.connectedComponents((dist >= level).astype(np.uint8))
        if all(core_labels[y, x] != core_labels[ky, kx]
               for _, kx, ky in separated):
            separated.append((strength, x, y))
    return separated


def detect(frame, blue_sat=165, peak_relative=.55, min_distance=9,
           colour_model=None, extra_excluded=None, hsv_config=None,
           active_colours=None):
    height, width = frame.shape[:2]
    scale = width / REFERENCE_WIDTH
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsv_config is not None:
        masks = configured_hsv_masks(frame, hsv_config)
    elif colour_model is not None:
        masks = trained_colour_masks(frame, colour_model)
    else:
        masks = colour_masks(hsv, blue_sat)
    if active_colours is not None:
        requested = tuple(active_colours)
        invalid = [colour for colour in requested if colour not in COLOURS]
        if invalid:
            raise ValueError(f'Unknown active colours: {invalid}')
        masks = {colour: masks[colour] for colour in requested}
        if not masks:
            raise ValueError('active_colours must not be empty')
    brightness = cv2.GaussianBlur(hsv[:, :, 2], (3, 3), 0)
    targets, excluded = detect_targets(masks, scale)
    if extra_excluded is not None:
        if extra_excluded.shape != excluded.shape:
            raise ValueError('extra_excluded must match the input frame size')
        excluded = cv2.bitwise_or(excluded, extra_excluded)
    gems = []
    # Opening removes isolated colour noise without expanding class masks.
    for colour, raw in masks.items():
        if hsv_config is not None:
            item = hsv_config['colours'][colour]
            config_scale = width / float(hsv_config.get('reference_width', width))
            min_area_limit = float(item['min_area']) * config_scale ** 2
            max_area_limit = float(item['max_area']) * config_scale ** 2
            kernel_size = max(1, round(float(item['morph_kernel']) * config_scale))
        else:
            min_area_limit = 35 * scale ** 2
            max_area_limit = 2500 * scale ** 2
            kernel_size = max(1, round(3 * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = raw.copy()
        mask[excluded > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               np.ones((kernel_size, kernel_size), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, bw, bh, area = map(int, stats[label])
            if area < min_area_limit or min(bw, bh) < 5 * scale:
                continue
            # Reject very large non-gem regions, e.g. a coloured field border.
            if area > max_area_limit:
                continue
            component = (labels[y:y+bh, x:x+bw] == label).astype(np.uint8) * 255
            # Printed labels, coloured borders and reflections can have enough
            # area but are only a stroke a few pixels thick.  A real gem must
            # contain a compact colour core, not merely a long/thin component.
            aspect = max(bw, bh) / max(1, min(bw, bh))
            fill_ratio = area / max(1, bw * bh)
            if aspect > 3.0 or fill_ratio < .22:
                continue
            peaks = blob_centers(component, min_distance * scale, peak_relative,
                                 require_neck=area <= 300 * scale**2)
            min_core_radius = max(2.5, 3.5 * scale)
            appearance_split = False
            # Check every eligible colour, without coordinates or a count
            # target. Require room for two minimum-size candidates.
            if len(peaks) == 1 and 70 * scale**2 <= area <= 300 * scale**2:
                refined = blob_centers(
                    component, min_distance * scale, peak_relative, True,
                    brightness[y:y+bh, x:x+bw])
                if len(refined) > 1:
                    peaks = refined
                    appearance_split = True
            peaks = [peak for peak in peaks if peak[0] >= min_core_radius]
            if not peaks:
                continue
            # Small touching gems may split too when their cores have a
            # clear neck. No target piece count is used anywhere.
            ambiguous = area > 300 * scale**2
            source = 'split_candidate' if len(peaks) > 1 else (
                'unresolved_blob' if ambiguous else 'single_candidate')
            if appearance_split:
                source = 'appearance_split_candidate'
                # Keep CSV peak_radius in pixel distance units, even though
                # brightness weighting was used to select the peak.
                radius_map = cv2.distanceTransform(
                    np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
                peaks = [(float(radius_map[py, px]), px, py)
                         for _, px, py in peaks]
            for strength, px, py in peaks:
                gems.append({'colour': colour, 'x': px+x, 'y': py+y,
                             'source': source, 'blob_area': area,
                             'peak_radius': strength})
    gems.sort(key=lambda gem: (gem['y'], gem['x'], gem['colour']))
    for i, gem in enumerate(gems, 1):
        gem['id'] = i
    return targets, gems, masks, excluded


def write_image(path, image):
    # imencode/tofile also supports Unicode Windows paths.
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    encoded.tofile(str(path))


def select_gems(gems, roi):
    """Keep detected centres inside a pile rectangle, without clipping blobs."""
    if roi is not None:
        x, y, width, height = roi
        gems = [g for g in gems if x <= g['x'] < x+width and y <= g['y'] < y+height]
    counts = dict.fromkeys(COLOURS, 0)
    result = []
    for i, gem in enumerate(gems, 1):
        colour = gem['colour']
        counts[colour] += 1
        result.append(dict(gem, id=i, colour_id=counts[colour],
                           label=f'{PREFIX[colour]}{counts[colour]:02d}'))
    return result


def fit_for_display(image, max_width=1200, max_height=700):
    h, w = image.shape[:2]
    ratio = min(1.0, max_width/w, max_height/h)
    return cv2.resize(image, (max(1, round(w*ratio)), max(1, round(h*ratio))),
                      interpolation=cv2.INTER_AREA) if ratio < 1 else image


def label_gem(image, point, gem, font=.36):
    """Compact colour code at each actual detected centre."""
    x, y = point
    colour = DISPLAY_COLOURS[gem['colour']]
    cv2.circle(image, point, 4, (255,255,255), 2)
    cv2.circle(image, point, 3, colour, -1)
    text = gem['label']
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    location = (max(0, min(x+6, image.shape[1]-tw-2)),
                max(th+2, min(y-5, image.shape[0]-3)))
    cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, font,
                (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, font,
                (15,15,15), 1, cv2.LINE_AA)


def colour_panels(frame, gems, roi):
    """Six separate views let users inspect dense piles one colour at a time."""
    h, w = frame.shape[:2]
    if roi is not None:
        x0, y0, cw, ch = roi
        x1, y1 = x0+cw, y0+ch
    elif gems:
        x0 = max(0, min(g['x'] for g in gems)-20)
        y0 = max(0, min(g['y'] for g in gems)-20)
        x1 = min(w, max(g['x'] for g in gems)+21)
        y1 = min(h, max(g['y'] for g in gems)+21)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    tile_w, tile_h = 520, 440
    scale = min((tile_w-20)/(x1-x0), (tile_h-65)/(y1-y0))
    cw, ch = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
    crop = cv2.resize(frame[y0:y1,x0:x1], (cw,ch), interpolation=cv2.INTER_LINEAR)
    panel = np.full((tile_h*2, tile_w*3, 3), 245, np.uint8)
    for i, colour in enumerate(COLOURS):
        tile = panel[(i//3)*tile_h:(i//3+1)*tile_h,
                     (i%3)*tile_w:(i%3+1)*tile_w]
        ox, oy = (tile_w-cw)//2, 55
        tile[oy:oy+ch,ox:ox+cw] = crop
        members = [g for g in gems if g['colour'] == colour]
        cv2.putText(tile, f'{PREFIX[colour]} = {colour} : {len(members)}', (18,32),
                    cv2.FONT_HERSHEY_SIMPLEX, .73, (25,25,25), 1, cv2.LINE_AA)
        for gem in members:
            point = (ox+round((gem['x']-x0)*cw/(x1-x0)),
                     oy+round((gem['y']-y0)*ch/(y1-y0)))
            label_gem(tile, point, gem, .43)
    return panel


def add_count_panel(overview, counts, target_count, roi, holes_enabled,
                    ignored_in_holes):
    height, width = overview.shape[:2]
    canvas = np.full((max(height, 470), width+285, 3), 245, np.uint8)
    canvas[:height, :width] = overview
    x = width+18
    def line(text, y, size=.53, colour=(30, 30, 30)):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, colour, 1, cv2.LINE_AA)
    line('GEM COLOURS - V4.1', 35, .62)
    line('Selected pile' if roi else 'Whole input image', 65)
    for i, colour in enumerate(COLOURS):
        y = 110+i*38
        cv2.circle(canvas, (x+8, y-6), 8, DISPLAY_COLOURS[colour], -1)
        cv2.putText(canvas, f'{PREFIX[colour]} {colour:6} {counts[colour]:3}', (x+28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .53, (30,30,30), 1, cv2.LINE_AA)
    line(f'ACTIVE: {sum(counts.values())}', 355, .7)
    line(f'Targets: {target_count}/6', 389)
    line(f'Hole mask: {"ON" if holes_enabled else "OFF"}', 423, .45)
    line(f'Ignored in holes: {ignored_in_holes}', 448, .43)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('gem_colors_output'))
    roi_options = parser.add_mutually_exclusive_group()
    roi_options.add_argument('--select-roi', action='store_true',
                             help='Draw rectangle around the pile; ENTER to accept')
    roi_options.add_argument('--roi', nargs=4, type=int, metavar=('X','Y','WIDTH','HEIGHT'))
    arena_options = parser.add_mutually_exclusive_group()
    arena_options.add_argument('--calibrate-arena', action='store_true',
                               help='Click four inner corners and save calibration')
    arena_options.add_argument('--arena-calibration', type=Path,
                               help='Reuse a saved arena_calibration.npz')
    parser.add_argument('--arena-output', type=Path,
                        default=Path('arena_calibration.npz'))
    parser.add_argument('--arena-width', type=int, default=1050,
                        help='Rectified width; default is 5 px/cm for 210 cm')
    parser.add_argument('--arena-height', type=int, default=600,
                        help='Rectified height; default is 5 px/cm for 120 cm')
    hole_options = parser.add_mutually_exclusive_group()
    hole_options.add_argument('--calibrate-holes', action='store_true',
                              help='After rectification, click centre + edge of six holes')
    hole_options.add_argument('--hole-calibration', type=Path,
                              help='Reuse a saved hole_calibration.npz')
    parser.add_argument('--hole-output', type=Path,
                        default=Path('hole_calibration.npz'))
    parser.add_argument('--hole-margin-cm', type=float, default=1.5,
                        help='Safety margin added around every calibrated hole')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--model', type=Path,
                        help='Trained gem_color_model.npz; omit for fixed HSV')
    parser.add_argument('--blue-sat', type=int, default=165,
                        help='Saturation boundary between light and dark blue')
    parser.add_argument('--peak-relative', type=float, default=.55)
    parser.add_argument('--min-distance', type=float, default=9,
                        help='Within-blob peak spacing at 741px image width')
    args = parser.parse_args()
    if args.select_roi and args.no_gui:
        parser.error('--select-roi needs a GUI; use --roi X Y WIDTH HEIGHT instead')
    if args.calibrate_arena and args.no_gui:
        parser.error('--calibrate-arena needs a GUI')
    if args.calibrate_holes and args.no_gui:
        parser.error('--calibrate-holes needs a GUI')
    if args.arena_width < 2 or args.arena_height < 2:
        parser.error('Arena output dimensions must be at least 2 pixels')
    if (not 1 <= args.blue_sat <= 255 or not 0 < args.peak_relative <= 1
            or args.min_distance <= 0 or args.hole_margin_cm < 0):
        parser.error('Invalid saturation, peak threshold, or spacing')
    if args.image is None:
        folder = Path(__file__).resolve().parent
        default = folder / 'arena_ref.png'
        alternatives = sorted(folder.glob('arena_ref*.png'))
        if default.exists():
            args.image = default
        elif len(alternatives) == 1:
            args.image = alternatives[0]
        else:
            parser.error('Specify the image: --image "arena_ref(1).png"')
    try:
        frame = cv2.imdecode(np.fromfile(str(args.image), np.uint8), cv2.IMREAD_COLOR)
    except OSError as exc:
        parser.error(f'Cannot read {args.image}: {exc}')
    if frame is None or min(frame.shape[:2]) < 2:
        parser.error(f'Invalid image: {args.image}')
    if args.calibrate_arena:
        try:
            corners = select_arena_corners(frame)
        except (cv2.error, ValueError) as exc:
            parser.error(f'Arena calibration failed: {exc}')
        if corners is None:
            print('Arena calibration cancelled. No results written.')
            return
        save_arena_calibration(args.arena_output, corners, frame.shape)
        normalized = corners / np.array(
            [frame.shape[1], frame.shape[0]], dtype=np.float32)
        frame = rectify_arena(
            frame, normalized, args.arena_width, args.arena_height)
        print(f'Arena rectified to {args.arena_width} x {args.arena_height}.')
    elif args.arena_calibration is not None:
        try:
            normalized = load_arena_calibration(args.arena_calibration)
            frame = rectify_arena(
                frame, normalized, args.arena_width, args.arena_height)
        except ValueError as exc:
            parser.error(str(exc))
        print('Arena calibration loaded:', args.arena_calibration.resolve())
    hole_centres = hole_radii = None
    if args.calibrate_holes:
        try:
            selected = select_holes(frame)
        except (cv2.error, ValueError) as exc:
            parser.error(f'Hole calibration failed: {exc}')
        if selected is None:
            print('Hole calibration cancelled. No results written.')
            return
        hole_centres, hole_radii = selected
        save_hole_calibration(args.hole_output, hole_centres, hole_radii)
    elif args.hole_calibration is not None:
        try:
            hole_centres, hole_radii = load_hole_calibration(
                args.hole_calibration)
        except ValueError as exc:
            parser.error(str(exc))
        print('Hole calibration loaded:', args.hole_calibration.resolve())
        from hole_guard import check_holes
        bad_holes = check_holes(frame, hole_centres, hole_radii)
        if bad_holes:
            parser.error('Hole calibration does not match image: '
                         + ', '.join(bad_holes) + '. Recalibrate holes first.')
    hole_mask = (make_hole_mask(frame.shape, hole_centres, hole_radii,
                                args.hole_margin_cm)
                 if hole_centres is not None else
                 np.zeros(frame.shape[:2], np.uint8))
    roi = args.roi
    if args.select_roi:
        # Fit to a typical laptop screen; convert selection back to source pixels.
        fh, fw = frame.shape[:2]
        ratio = min(1.0, 1200/fw, 700/fh)
        preview = cv2.resize(frame, (round(fw*ratio), round(fh*ratio)))
        window = 'Select pile - drag rectangle then ENTER (C cancels)'
        try:
            rx, ry, rw, rh = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window)
        except cv2.error:
            parser.error('ROI window unavailable; use --roi X Y WIDTH HEIGHT')
        if rw == 0 or rh == 0:
            print('Selection cancelled. No results written.')
            return
        x, y = round(rx*fw/preview.shape[1]), round(ry*fh/preview.shape[0])
        right = min(fw, round((rx+rw)*fw/preview.shape[1]))
        bottom = min(fh, round((ry+rh)*fh/preview.shape[0]))
        roi = (x, y, right-x, bottom-y)
    if roi is not None:
        x, y, rw, rh = roi
        if x < 0 or y < 0 or rw <= 0 or rh <= 0 or x+rw > frame.shape[1] or y+rh > frame.shape[0]:
            parser.error('ROI must be a positive rectangle inside the input image')
    colour_model = None
    if args.model is not None:
        try:
            colour_model = load_colour_model(args.model)
        except ValueError as exc:
            parser.error(str(exc))
        print('Colour mode: trained LAB model -', args.model.resolve())
    else:
        print('Colour mode: original fixed HSV fallback')
    targets, gems, _, _ = detect(
        frame, args.blue_sat, args.peak_relative, args.min_distance,
        colour_model)
    # Decide delivery using the detected gem centre. A gem may touch a hole
    # boundary while being carried; it is ignored only after its centre enters
    # the calibrated no-count zone.
    ignored_in_holes = [g for g in gems if hole_mask[g['y'], g['x']] > 0]
    gems = [g for g in gems if hole_mask[g['y'], g['x']] == 0]
    gems = select_gems(gems, roi)
    counts = {c: sum(g['colour'] == c for g in gems) for c in COLOURS}
    h, w = frame.shape[:2]
    overview = frame.copy()
    if hole_centres is not None:
        draw_hole_zones(overview, hole_centres, hole_radii,
                        args.hole_margin_cm)
    for colour, target in targets.items():
        cx, cy = target['center']
        cv2.drawContours(overview, [target['contour']], -1, (255, 255, 255), 2)
        cv2.putText(overview, colour, (max(0, cx-25), max(18, cy-40)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    for gem in gems:
        p = (gem['x'], gem['y'])
        label_gem(overview, p, gem)
    if roi is not None:
        x, y, rw, rh = roi
        cv2.rectangle(overview, (x,y), (x+rw-1,y+rh-1), (0,220,220), 2)
    cv2.putText(overview, f"Targets {len(targets)}/6 | Active gems: {len(gems)}",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
    # Zoom bounds are derived from detections, never fixed field coordinates.
    zoom = None
    if gems:
        x0 = max(0, min(g['x'] for g in gems)-20)
        y0 = max(0, min(g['y'] for g in gems)-20)
        x1 = min(w, max(g['x'] for g in gems)+21)
        y1 = min(h, max(g['y'] for g in gems)+21)
        z = min(3.0, 1100/(x1-x0), 650/(y1-y0))
        zoom = cv2.resize(frame[y0:y1, x0:x1], None, fx=z, fy=z,
                          interpolation=cv2.INTER_NEAREST)
        for gem in gems:
            p = (round((gem['x']-x0)*z), round((gem['y']-y0)*z))
            label_gem(zoom, p, gem, .40)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = add_count_panel(overview, counts, len(targets), roi,
                               hole_centres is not None,
                               len(ignored_in_holes))
    write_image(args.output_dir/'overview.png', overview)
    panels = colour_panels(frame, gems, roi)
    write_image(args.output_dir/'gem_colors.png', panels)
    if zoom is not None:
        write_image(args.output_dir/'gems_zoom.png', zoom)
    elif (args.output_dir/'gems_zoom.png').exists():
        # Replace an old populated zoom so an empty result cannot show stale gems.
        empty = np.full((100,400,3), 245, np.uint8)
        cv2.putText(empty, 'No gems in selection', (15,55), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0,0,0), 1, cv2.LINE_AA)
        write_image(args.output_dir/'gems_zoom.png', empty)
    with (args.output_dir/'counts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['colour', 'active_candidate_count'])
        writer.writerows(counts.items())
        writer.writerow(['TOTAL', len(gems)])
    with (args.output_dir/'gems.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['id', 'label', 'colour_id', 'colour', 'state', 'x', 'y', 'approx_x_cm', 'approx_y_cm', 'source', 'blob_area', 'peak_radius']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for gem in gems:
            writer.writerow(dict(gem, state='ACTIVE',
                                 approx_x_cm=round(gem['x']/(w-1)*210, 2),
                                 approx_y_cm=round(gem['y']/(h-1)*120, 2)))
    print('Targets detected:', len(targets), '/ 6')
    print('Count scope:', 'selected pile' if roi else 'whole input image')
    print('Active gem candidates:', len(gems))
    print('Ignored inside calibrated holes:', len(ignored_in_holes))
    for colour in COLOURS:
        print(f'{colour}:', sum(g['colour'] == colour for g in gems))
    print('cm values are approximate; full image assumed 210 x 120 cm.')
    print('Saved results:', args.output_dir.resolve())
    if not args.no_gui:
        try:
            # One window avoids hidden dialogs; switch views with 1/2/3.
            window = 'Gem Colours V4.1 | 1 Overview  2 By colour  3 Zoom | Q Quit'
            views = {ord('1'): overview, ord('2'): panels,
                     ord('3'): zoom if zoom is not None else panels}
            active = ord('1')
            cv2.imshow(window, fit_for_display(views[active]))
            print('Viewing results: press 1/2/3 to switch, Q/Esc or close window to exit.')
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in views:
                    active = key
                    cv2.imshow(window, fit_for_display(views[active]))
        except cv2.error:
            print('GUI unavailable. Open the saved PNG files instead.')
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
