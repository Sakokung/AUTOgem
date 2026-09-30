"""Continuous camera-feedback pickup and colour-sorting controller."""
import math
import threading
import time


def match_target(gems, locked, max_shift=30):
    matches = [gem for gem in gems if gem['colour'] == locked['colour']
               and (gem['x']-locked['x'])**2 + (gem['y']-locked['y'])**2
               <= max_shift**2]
    return min(matches, key=lambda gem:
               (gem['x']-locked['x'])**2 + (gem['y']-locked['y'])**2) if matches else None


def gem_in_gripper(robot, gem, max_tip_distance_scale=.72,
                   max_lateral_scale=.62):
    """Geometrically check whether a gem is near the gripper capture zone.

    This is useful for planning/tests, but auto-close additionally requires a
    live detection inside the camera ROI.  Limits scale with the ArUco size.
    """
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    forward = math.hypot(fx, fy)
    marker_side = float(robot['marker_side_px'])
    if forward < 8 or marker_side <= 0:
        return False
    ux, uy = fx/forward, fy/forward
    dx, dy = gem['x']-tip_x, gem['y']-tip_y
    along = dx*ux + dy*uy
    lateral = abs(ux*dy-uy*dx)
    tip_distance = math.hypot(dx, dy)
    return (tip_distance <= max(28.0, max_tip_distance_scale*marker_side)
            and abs(along) <= .72*marker_side
            and lateral <= max_lateral_scale*marker_side)


def plan_step(robot, gem, captured=False):
    """Return (command, pulse seconds, distance px, signed angle degrees)."""
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    dx, dy = gem['x']-tip_x, gem['y']-tip_y
    forward = math.hypot(fx, fy)
    distance = math.hypot(dx, dy)
    if forward < 8 or not math.isfinite(distance):
        raise ValueError('พิกัดปลาย Gripper ไม่ชัด')
    angle = math.degrees(math.atan2(fx*dy-fy*dx, fx*dx+fy*dy))
    stop_distance = max(28.0, .72*robot['marker_side_px'])
    if captured:
        return 'GRIP', 0, distance, angle
    if abs(angle) > 12:
        return ('R' if angle > 0 else 'L'), .12, distance, angle
    return 'F', (.12 if distance < 2*stop_distance else .18), distance, angle


def plan_delivery_step(robot, hole):
    """Drive the gripper tip into a colour hole, then request a drop."""
    destination = {'x': hole['x'], 'y': hole['y']}
    _, duration, distance, angle = plan_step(robot, destination)
    drop_distance = max(12.0, .55 * float(hole['radius']))
    if distance <= drop_distance:
        return 'DROP', 0, distance, angle
    if abs(angle) > 12:
        return ('R' if angle > 0 else 'L'), .12, distance, angle
    return 'F', (.10 if distance < 2.2*drop_distance else .18), distance, angle


class ApproachRunner:
    def __init__(self, snapshot, send, stop, ready):
        self.snapshot = snapshot
        self.send = send
        self.stop_motors = stop
        self.ready = ready
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.running = False
        self.message = 'ยังไม่เริ่ม'
        self.pulses = 0
        self.target = None

    def state(self):
        with self.lock:
            return dict(running=self.running, message=self.message,
                        pulses=self.pulses, target=self.target)

    def _message(self, value):
        with self.lock:
            self.message = value

    def start(self):
        with self.lock:
            if self.running:
                raise ValueError('รถกำลังวิ่งไปหาเป้าหมายอยู่')
            if not self.ready():
                raise ValueError('ต้องเปิดกล้องจริง อยู่แท็บควบคุมรถ และปิดการควบคุมด้วยปุ่ม')
            observation = self.snapshot()
            if (observation is None or observation['robot'] is None
                    or observation['target'] is None
                    or time.monotonic()-observation['time'] > .8):
                raise ValueError('ยังไม่พบ ArUco ID 0 คันเดียวและหินเป้าหมายในภาพล่าสุด')
            target_colour = observation['target']['colour']
            if target_colour not in observation.get('holes', {}):
                raise ValueError(f'ยังไม่ได้ตั้งวงรับหินสี {target_colour}')
            self.target = {key: observation['target'][key]
                           for key in ('colour', 'x', 'y')}
            self.running = True
            self.pulses = 0
            self.message = 'กำลังยืนยันหินเป้าหมาย'
            self.cancel.clear()
            threading.Thread(target=self._run, daemon=True).start()

    def abort(self):
        self.cancel.set()
        self.stop_motors()
        self._message('หยุดตามคำสั่ง')

    def _send_continuous(self, packet):
        """Retry transient readiness/link failures until cancelled."""
        while not self.cancel.is_set():
            try:
                self.send(packet)
                return True
            except (ValueError, ConnectionError) as exc:
                self.stop_motors()
                self._message(f'{exc}; กำลังเชื่อมต่อแล้วลองใหม่')
                self.cancel.wait(.5)
        return False

    def _run(self):
        last_frame = -1
        stable = 0
        no_progress = 0
        previous_pose = None
        previous_turn_error = None
        wrong_turns = 0
        capture_frames = 0
        phase = 'pickup'
        try:
            self._message('เปิดกริปเปอร์เพื่อเตรียมรับหิน')
            if not self._send_continuous('O'):
                return
            if self.cancel.wait(.45):
                return
            while not self.cancel.is_set():
                if not self.ready():
                    self.stop_motors()
                    self._message('รอกล้องหรือโหมดควบคุมกลับมาพร้อม')
                    self.cancel.wait(.25)
                    continue
                observation = self.snapshot()
                if observation is None or time.monotonic()-observation['time'] > .8:
                    self.stop_motors()
                    self._message('รอภาพกล้องอัปเดต')
                    self.cancel.wait(.10)
                    continue
                if observation['frame_index'] == last_frame:
                    self.cancel.wait(.025)
                    continue
                last_frame = observation['frame_index']
                robot = observation['robot']
                if robot is None:
                    self.stop_motors()
                    self._message('รอให้เห็น ArUco ID 0 เพียงคันเดียว')
                    self.cancel.wait(.10)
                    continue
                if self.target is None:
                    candidate = observation.get('target')
                    if candidate is None:
                        self.stop_motors()
                        self._message('รอหินก้อนถัดไป')
                        self.cancel.wait(.10)
                        continue
                    if candidate['colour'] not in observation.get('holes', {}):
                        self.stop_motors()
                        self._message(f'รอการตั้งวงรับหินสี {candidate["colour"]}')
                        self.cancel.wait(.25)
                        continue
                    self.target = {key: candidate[key]
                                   for key in ('colour', 'x', 'y')}
                    stable = 0
                    capture_frames = 0
                    previous_pose = None
                    previous_turn_error = None
                    wrong_turns = 0
                    no_progress = 0
                    self._message(f'เลือกหินก้อนใหม่สี {self.target["colour"]}')
                if phase == 'deliver':
                    hole = observation.get('holes', {}).get(self.target['colour'])
                    if hole is None:
                        self.stop_motors()
                        self._message(f'รอวงรับหินสี {self.target["colour"]}')
                        self.cancel.wait(.25)
                        continue
                    current_pose = (robot['x'], robot['y'], robot['heading_deg'])
                    if previous_pose is not None:
                        dx = current_pose[0]-previous_pose[0]
                        dy = current_pose[1]-previous_pose[1]
                        turn = ((current_pose[2]-previous_pose[2]+180) % 360)-180
                        no_progress = (no_progress+1 if math.hypot(dx, dy) < 2
                                       and abs(turn) < 3 else 0)
                        if no_progress >= 3:
                            self.stop_motors()
                            self._message('รถยังไม่ขยับ; รอแล้วลองต่ออัตโนมัติ')
                            previous_pose = None
                            no_progress = 0
                            self.cancel.wait(.5)
                            continue
                    command, duration, distance, angle = plan_delivery_step(robot, hole)
                    if previous_turn_error is not None:
                        wrong_turns = (wrong_turns+1
                                       if abs(angle) > previous_turn_error+5 else 0)
                        if wrong_turns >= 2:
                            self.stop_motors()
                            self._message('มุมห่างวงสีเพิ่ม; ตั้งหลักแล้วลองต่อ')
                            previous_turn_error = None
                            wrong_turns = 0
                            self.cancel.wait(.35)
                            continue
                    if command == 'DROP':
                        self.stop_motors()
                        self._message(f'ถึงวงสี {self.target["colour"]}; กำลังปล่อยหิน')
                        if not self._send_continuous('O'):
                            return
                        self.cancel.wait(.7)
                        completed_colour = self.target['colour']
                        self.target = None
                        phase = 'pickup'
                        stable = 0
                        capture_frames = 0
                        previous_pose = None
                        previous_turn_error = None
                        wrong_turns = 0
                        no_progress = 0
                        self._message(
                            f'วางหินในวงสี {completed_colour} แล้ว; รอหินก้อนถัดไป')
                        continue
                    self._message(
                        f'พาหินไปวง {self.target["colour"]}: {command} | '
                        f'ห่าง {distance:.0f} px | มุม {angle:.0f}°')
                    previous_pose = current_pose
                    previous_turn_error = abs(angle) if command in ('L', 'R') else None
                    if not self._send_continuous(command):
                        return
                    self.pulses += 1
                    self.cancel.wait(duration)
                    self.stop_motors()
                    continue
                gem = match_target(observation['gems'], self.target)
                if gem is None:
                    # A gem can disappear behind the gripper as the car gets
                    # close. Continue toward the last confirmed coordinates.
                    gem = self.target
                    self._message('มองไม่เห็นเป้าหมาย; ไปยังพิกัดล่าสุดต่อ')
                else:
                    self.target = {key: gem[key] for key in ('colour', 'x', 'y')}
                captured_gem = match_target(
                    observation.get('gripper_gems', []), self.target)
                capture_frames = capture_frames + 1 if captured_gem is not None else 0
                if stable < 3:
                    stable += 1
                    self._message(f'ยืนยันเป้าหมาย {stable}/3 เฟรม')
                    continue
                current_pose = (robot['x'], robot['y'], robot['heading_deg'])
                if previous_pose is not None:
                    dx = current_pose[0]-previous_pose[0]
                    dy = current_pose[1]-previous_pose[1]
                    turn = ((current_pose[2]-previous_pose[2]+180) % 360)-180
                    no_progress = no_progress+1 if math.hypot(dx,dy)<2 and abs(turn)<3 else 0
                    if no_progress >= 3:
                        self.stop_motors()
                        self._message('รถยังไม่ขยับ; รอแล้วลองต่ออัตโนมัติ')
                        previous_pose = None
                        no_progress = 0
                        self.cancel.wait(.5)
                        continue
                # Require two consecutive camera frames with the target centre
                # inside the jaw ROI before closing the gripper.
                command, duration, distance, angle = plan_step(
                    robot, captured_gem or gem, captured=capture_frames >= 2)
                if previous_turn_error is not None:
                    wrong_turns = (wrong_turns+1 if abs(angle) > previous_turn_error+5
                                   else 0)
                    if wrong_turns >= 2:
                        self.stop_motors()
                        self._message('มุมห่างเป้าหมายเพิ่ม; ตั้งหลักแล้วลองต่อ')
                        previous_turn_error = None
                        wrong_turns = 0
                        self.cancel.wait(.35)
                        continue
                if command == 'GRIP':
                    self.stop_motors()
                    self._message(f'พบหินในกริปเปอร์ {distance:.0f} px; กำลังคีบ')
                    if not self._send_continuous('C'):
                        return
                    self.cancel.wait(.7)
                    self._message(f'คีบหินแล้ว; กำลังไปวงสี {self.target["colour"]}')
                    phase = 'deliver'
                    previous_pose = None
                    previous_turn_error = None
                    wrong_turns = 0
                    no_progress = 0
                    continue
                self._message(f'{command} สั้น ๆ | ห่าง {distance:.0f} px | มุม {angle:.0f}°')
                previous_pose = current_pose
                previous_turn_error = abs(angle) if command in ('L', 'R') else None
                if not self._send_continuous(command):
                    return
                self.pulses += 1
                self.cancel.wait(duration)
                self.stop_motors()
        except (ValueError, ConnectionError) as exc:
            # Unexpected command/geometry errors still leave the motors safe.
            # The outer controller can be started again after the cause is fixed.
            self._message(str(exc))
        finally:
            self.stop_motors()
            with self.lock:
                self.running = False
