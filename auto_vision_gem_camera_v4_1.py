"""Live V4.1: ArUco car/gripper mask + arena + holes + frame confirmation.

Requires: pip install opencv-contrib-python numpy
Create arena_calibration.npz and hole_calibration.npz with the image program
before running this live program. Press S to save a snapshot and Q/Esc to quit.
"""
import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from auto_vision_gem_colors_v4_1 import (
    COLOURS, DISPLAY_COLOURS, PREFIX, detect, draw_hole_zones,
    fit_for_display, load_arena_calibration, load_colour_model,
    load_hole_calibration, load_hsv_config, make_hole_mask,
    rectify_arena, select_gems,
    write_image,
)


# The live controller activates one entry at a time: GREEN first, then CYAN.
AUTO_COLOUR_ORDER = ("GREEN", "CYAN")


def local_to_image(center, forward, right, marker_side, local_points):
    """Convert marker-relative (forward, right) coordinates to image pixels."""
    points = []
    for ahead, lateral in local_points:
        point = (center + marker_side *
                 (ahead * forward + lateral * right))
        points.append(np.rint(point).astype(np.int32))
    return np.asarray(points, dtype=np.int32)


def detect_aruco(frame, rear_scale=1.35, body_front_scale=1.55,
                 body_half_width_scale=1.35, gripper_length_scale=3.65,
                 gripper_half_width_scale=1.65):
    """Detect DICT_4X4_50 and build an oriented body + gripper mask.

    Scale defaults were measured from Sky's six 1920x1080 field snapshots.
    The marker's canonical top edge points toward the physical gripper.
    """
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError(
            'OpenCV ArUco is unavailable. Install: pip install opencv-contrib-python')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = (cv2.aruco.DetectorParameters()
                  if hasattr(cv2.aruco, 'DetectorParameters')
                  else cv2.aruco.DetectorParameters_create())
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 53
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(
            dictionary, parameters).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters)
    car_mask = np.zeros(gray.shape, np.uint8)
    markers = []
    if ids is None:
        return markers, car_mask
    for sequence, (marker_corners, marker_id) in enumerate(
            zip(corners, ids.flatten()), 1):
        points = marker_corners.reshape(4, 2).astype(np.float32)
        center_f = points.mean(axis=0)
        front_edge = (points[0] + points[1]) / 2.0
        forward = front_edge - center_f
        forward /= max(float(np.linalg.norm(forward)), 1e-6)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        sides = [np.linalg.norm(points[(index + 1) % 4] - points[index])
                 for index in range(4)]
        marker_side = float(np.mean(sides))
        body_local = [
            (-rear_scale, -body_half_width_scale),
            (body_front_scale, -body_half_width_scale),
            (body_front_scale, body_half_width_scale),
            (-rear_scale, body_half_width_scale),
        ]
        # Cover the whole gripper first, then uncover only the narrow space
        # between its arms.  This keeps coloured hardware out of normal gem
        # detection while allowing the camera to see a gem in the jaws.
        gripper_local = [
            (1.10, -.78),
            (gripper_length_scale, -gripper_half_width_scale),
            (gripper_length_scale, gripper_half_width_scale),
            (1.10, .78),
        ]
        body_polygon = local_to_image(
            center_f, forward, right, marker_side, body_local)
        gripper_polygon = local_to_image(
            center_f, forward, right, marker_side, gripper_local)
        cv2.fillPoly(car_mask, [body_polygon, gripper_polygon], 255)
        gripper_capture_local = [
            (1.45, -.75),
            (3.75, -.75),
            (3.75, .75),
            (1.45, .75),
        ]
        gripper_capture_polygon = local_to_image(
            center_f, forward, right, marker_side, gripper_capture_local)
        cv2.fillPoly(car_mask, [gripper_capture_polygon], 0)
        tip_f = center_f + marker_side * gripper_length_scale * forward
        dx, dy = forward
        heading = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
        markers.append({
            'track': sequence,
            'aruco_id': int(marker_id),
            'x': int(round(center_f[0])),
            'y': int(round(center_f[1])),
            'heading_deg': float(heading),
            'marker_side_px': marker_side,
            'corners': np.rint(points).astype(np.int32),
            'front': tuple(np.rint(front_edge).astype(int)),
            'gripper_tip_x': int(round(tip_f[0])),
            'gripper_tip_y': int(round(tip_f[1])),
            'body_polygon': body_polygon,
            'gripper_polygon': gripper_polygon,
            'gripper_capture_polygon': gripper_capture_polygon,
        })
    return markers, car_mask


class FrameConfirmer:
    """Accept only detections matched uniquely across consecutive frames."""

    def __init__(self, frames=3, max_distance_px=15.0):
        self.frames = frames
        self.max_distance_sq = max_distance_px ** 2
        self.history = deque(maxlen=max(0, frames - 1))

    def update(self, current):
        if self.frames == 1:
            return list(current)
        stable = set(range(len(current)))
        for old in self.history:
            edges = []
            for ci, new_gem in enumerate(current):
                for oi, old_gem in enumerate(old):
                    if new_gem['colour'] != old_gem['colour']:
                        continue
                    distance_sq = ((new_gem['x'] - old_gem['x']) ** 2
                                   + (new_gem['y'] - old_gem['y']) ** 2)
                    if distance_sq <= self.max_distance_sq:
                        edges.append((distance_sq, ci, oi))
            matched_current, matched_old = set(), set()
            for _distance, ci, oi in sorted(edges):
                if ci not in matched_current and oi not in matched_old:
                    matched_current.add(ci)
                    matched_old.add(oi)
            stable.intersection_update(matched_current)
        ready = len(self.history) == self.frames - 1
        confirmed = [gem for index, gem in enumerate(current)
                     if ready and index in stable]
        self.history.append([dict(gem) for gem in current])
        return confirmed


def open_camera(index, backend, width, height):
    backend_codes = {
        'auto': cv2.CAP_ANY,
        'dshow': getattr(cv2, 'CAP_DSHOW', cv2.CAP_ANY),
        'msmf': getattr(cv2, 'CAP_MSMF', cv2.CAP_ANY),
    }
    attempts = [backend_codes[backend]]
    if attempts[0] != cv2.CAP_ANY:
        attempts.append(cv2.CAP_ANY)
    for backend_code in attempts:
        camera = cv2.VideoCapture(index, backend_code)
        if not camera.isOpened():
            camera.release()
            continue
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            ok, frame = camera.read()
            if ok and frame is not None:
                return camera
        camera.release()
    return None


def target_route(robot, gem):
    """Return distance, shortest signed turn, and path cost to one gem."""
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    dx, dy = gem['x']-tip_x, gem['y']-tip_y
    forward = float(np.hypot(fx, fy))
    distance = float(np.hypot(dx, dy))
    marker_side = max(1.0, float(robot.get('marker_side_px', 1.0)))
    if forward < 1e-6:
        return distance, 0.0, float('inf')
    angle = float(np.degrees(np.arctan2(fx*dy-fy*dx, fx*dx+fy*dy)))
    # Approximate the time cost of turning: every 45 degrees counts as one
    # marker width of forward travel. This avoids a close target behind the
    # robot winning over a slightly farther target already in front of it.
    path_cost = distance + marker_side*abs(angle)/45.0
    return distance, angle, path_cost


def choose_nearest_target(markers, gems, robot_id=0,
                          colour_order=AUTO_COLOUR_ORDER):
    """Pick the nearest gem from the first available colour in the sequence.

    No target is returned if the requested marker is missing or ambiguous.
    This function only chooses a visual target; it never sends motor commands.
    """
    robots = [marker for marker in markers if marker['aruco_id'] == robot_id]
    if len(robots) != 1 or not gems:
        return None
    robot = robots[0]
    priorities = {colour: index for index, colour in enumerate(colour_order)}
    gems = [item for item in gems if item['colour'] in priorities]
    if not gems:
        return None
    first_priority = min(priorities.get(item['colour'], len(priorities))
                         for item in gems)
    current_colour_gems = [
        item for item in gems
        if priorities.get(item['colour'], len(priorities)) == first_priority
    ]
    gem = min(current_colour_gems, key=lambda item: (
        target_route(robot, item)[2],
        target_route(robot, item)[0],
        item['colour'], item['x'], item['y']))
    distance_px, _, _ = target_route(robot, gem)
    return robot, gem, distance_px


def annotate(frame, targets, pending, confirmed, markers, centres, radii,
             margin, fps, confirm_frames, target_aruco_id=0):
    result = frame.copy()
    draw_hole_zones(result, centres, radii, margin)
    for marker in markers:
        cv2.polylines(result, [marker['body_polygon']], True,
                      (0, 210, 255), 2, cv2.LINE_AA)
        cv2.polylines(result, [marker['gripper_polygon']], True,
                      (0, 165, 255), 2, cv2.LINE_AA)
        cv2.polylines(result, [marker['corners'].reshape(-1, 1, 2)], True,
                      (0, 255, 0), 2, cv2.LINE_AA)
        centre = (marker['x'], marker['y'])
        tip = (marker['gripper_tip_x'], marker['gripper_tip_y'])
        cv2.arrowedLine(result, centre, tip, (0, 255, 255), 2,
                        cv2.LINE_AA, tipLength=.12)
        label = (f"ArUco {marker['aruco_id']}  "
                 f"{marker['heading_deg']:.0f} deg")
        cv2.putText(result, label,
                    (max(0, centre[0] - 55), max(85, centre[1] - 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.putText(result, label,
                    (max(0, centre[0] - 55), max(85, centre[1] - 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 90, 0), 1,
                    cv2.LINE_AA)
    for target in targets.values():
        cv2.drawContours(result, [target['contour']], -1,
                         (255, 255, 255), 1)
    for gem in pending:
        cv2.circle(result, (gem['x'], gem['y']), 2, (120, 120, 120), -1)
    for gem in confirmed:
        point = (gem['x'], gem['y'])
        colour = DISPLAY_COLOURS[gem['colour']]
        cv2.circle(result, point, 5, (255, 255, 255), 2)
        cv2.circle(result, point, 3, colour, -1)
        cv2.putText(result, gem['label'], (point[0] + 6, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1,
                    cv2.LINE_AA)
    selected = choose_nearest_target(markers, confirmed, target_aruco_id)
    if selected is not None:
        robot, gem, distance_px = selected
        tip = (robot['gripper_tip_x'], robot['gripper_tip_y'])
        point = (gem['x'], gem['y'])
        cv2.line(result, tip, point, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(result, point, 15, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(result, f"TARGET {gem['label']} {distance_px:.0f}px",
                    (max(0, point[0]-35), max(85, point[1]-20)),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, f"TARGET {gem['label']} {distance_px:.0f}px",
                    (max(0, point[0]-35), max(85, point[1]-20)),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
    counts = {colour: sum(g['colour'] == colour for g in confirmed)
              for colour in COLOURS}
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (result.shape[1], 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, .58, result, .42, 0, result)
    status = (f'ArUco {len(markers)} | ACTIVE {len(confirmed)} | pending {len(pending)} | '
              f'confirm {confirm_frames} frames | FPS {fps:.1f}')
    count_text = '  '.join(f'{PREFIX[c]}:{counts[c]}' for c in COLOURS)
    cv2.putText(result, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, count_text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, 'S: save   Q/Esc: quit',
                (10, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                .50, (255, 255, 255), 2, cv2.LINE_AA)
    return result, counts


def save_snapshot(output_dir, frame, overview, gems, markers, counts):
    folder = output_dir / time.strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / 'arena_rectified.png', frame)
    write_image(folder / 'active_gems.png', overview)
    height, width = frame.shape[:2]
    with (folder / 'gems.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['id', 'label', 'colour_id', 'colour', 'state', 'x', 'y',
                  'approx_x_cm', 'approx_y_cm', 'source', 'blob_area',
                  'peak_radius']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for gem in gems:
            writer.writerow(dict(gem, state='ACTIVE',
                                 approx_x_cm=round(gem['x']/(width-1)*210, 2),
                                 approx_y_cm=round(gem['y']/(height-1)*120, 2)))
    with (folder / 'counts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['colour', 'active_confirmed_count'])
        writer.writerows((colour, counts[colour]) for colour in COLOURS)
        writer.writerow(['TOTAL', len(gems)])
    with (folder / 'cars.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['track', 'aruco_id', 'x', 'y', 'heading_deg',
                  'marker_side_px', 'gripper_tip_x', 'gripper_tip_y']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for marker in markers:
            writer.writerow({field: marker[field] for field in fields})
    print('Saved snapshot:', folder.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=('auto', 'dshow', 'msmf'),
                        default='dshow')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--arena-calibration', type=Path, required=True)
    parser.add_argument('--hole-calibration', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('gem_color_model.npz'))
    parser.add_argument('--hsv-config', type=Path,
                        help='Use hsv_colour_config.json instead of the LAB model')
    parser.add_argument('--arena-width', type=int, default=1050)
    parser.add_argument('--arena-height', type=int, default=600)
    parser.add_argument('--hole-margin-cm', type=float, default=1.5)
    parser.add_argument('--confirm-frames', type=int, default=3)
    parser.add_argument('--target-aruco-id', type=int, default=0,
                        help='Robot marker for previewing the nearest gem')
    parser.add_argument('--match-distance-cm', type=float, default=3.0)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('vision_camera_output'))
    parser.add_argument('--peak-relative', type=float, default=.55)
    parser.add_argument('--min-distance', type=float, default=9)
    parser.add_argument('--car-rear-scale', type=float, default=1.35,
                        help='Body distance behind marker, in marker sides')
    parser.add_argument('--body-front-scale', type=float, default=1.55,
                        help='Body distance toward gripper, in marker sides')
    parser.add_argument('--body-half-width-scale', type=float, default=1.35)
    parser.add_argument('--gripper-length-scale', type=float, default=3.65)
    parser.add_argument('--gripper-half-width-scale', type=float, default=1.65)
    args = parser.parse_args()
    if (args.camera < 0 or args.width <= 0 or args.height <= 0
            or args.arena_width < 2 or args.arena_height < 2
            or args.hole_margin_cm < 0 or args.confirm_frames < 1
            or args.match_distance_cm <= 0 or args.target_aruco_id < 0
            or min(args.car_rear_scale, args.body_front_scale,
                   args.body_half_width_scale, args.gripper_length_scale,
                   args.gripper_half_width_scale) <= 0):
        parser.error('Invalid camera, dimensions, margin, or confirmation settings')
    try:
        arena = load_arena_calibration(args.arena_calibration)
        centres, radii = load_hole_calibration(args.hole_calibration)
        hsv_config = (load_hsv_config(args.hsv_config)
                      if args.hsv_config is not None else None)
        model = (None if hsv_config is not None
                 else load_colour_model(args.model))
    except ValueError as exc:
        parser.error(str(exc))
    camera = open_camera(args.camera, args.backend, args.width, args.height)
    if camera is None:
        parser.error(f'Cannot open camera {args.camera}; close Camera/OBS or try another index')
    px_per_cm = ((args.arena_width - 1) / 210.0
                 + (args.arena_height - 1) / 120.0) / 2.0
    confirmer = FrameConfirmer(args.confirm_frames,
                               args.match_distance_cm * px_per_cm)
    hole_mask = make_hole_mask((args.arena_height, args.arena_width),
                               centres, radii, args.hole_margin_cm)
    window = 'Gem Camera V4.1 | ArUco Mask | S Save | Q Quit'
    fps, last_time = 0.0, time.perf_counter()
    mode = 'HSV slider config' if hsv_config is not None else 'trained LAB model'
    print(f'Live detection started with {mode}. Grey points are pending; labelled points are confirmed.')
    try:
        while True:
            ok, raw = camera.read()
            if not ok or raw is None:
                print('Camera frame could not be read.')
                break
            frame = rectify_arena(raw, arena, args.arena_width, args.arena_height)
            markers, car_mask = detect_aruco(
                frame, args.car_rear_scale, args.body_front_scale,
                args.body_half_width_scale, args.gripper_length_scale,
                args.gripper_half_width_scale)
            targets, detected, _, _ = detect(
                frame, peak_relative=args.peak_relative,
                min_distance=args.min_distance, colour_model=model,
                extra_excluded=car_mask, hsv_config=hsv_config)
            active = [gem for gem in detected
                      if hole_mask[gem['y'], gem['x']] == 0]
            confirmed = select_gems(confirmer.update(active), None)
            now = time.perf_counter()
            instant = 1.0 / max(now - last_time, 1e-6)
            fps = instant if fps == 0 else .85 * fps + .15 * instant
            last_time = now
            overview, counts = annotate(
                frame, targets, active, confirmed, markers, centres, radii,
                args.hole_margin_cm, fps, args.confirm_frames,
                args.target_aruco_id)
            cv2.imshow(window, fit_for_display(overview))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('s'), ord('S')):
                save_snapshot(args.output_dir, frame, overview,
                              confirmed, markers, counts)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error as exc:
        print('OpenCV camera/window error:', exc)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print('Camera released.')


if __name__ == '__main__':
    main()
