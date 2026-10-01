"""Continuous camera-feedback pickup and colour-sorting controller."""
import math
import statistics
import threading
import time


# Recalculate from live camera frames for this long before the next small move.
FEEDBACK_DELAY_SECONDS = 1.5
TURN_ENTER_DEGREES = 12
TURN_EXIT_DEGREES = 8
TURN_MIN_PULSE_SECONDS = .025
TURN_MAX_PULSE_SECONDS = .055
FORWARD_MIN_PULSE_SECONDS = .03
FORWARD_MAX_PULSE_SECONDS = .10
BACK_AWAY_PULSE_SECONDS = 3.0
MAX_MOTOR_PULSE_SECONDS = .5
DROP_CENTER_RADIUS_RATIO = .20
DROP_CENTER_MIN_PX = 2.0
DROP_CENTER_MAX_PX = 5.0
AUTO_COLOUR_SEQUENCE = ('GREEN', 'CYAN')


def target_copy(gem):
    """Copy only the fields needed to navigate toward a detected stone."""
    keys = ('colour', 'x', 'y')
    return {key: gem[key] for key in keys if key in gem}


def _scaled_pulse(error, near, far, minimum, maximum):
    """Scale a pulse continuously and keep it inside usable motor limits."""
    if far <= near:
        return minimum
    ratio = max(0.0, min(1.0, (error-near)/(far-near)))
    return minimum + ratio*(maximum-minimum)


def alignment_limits(distance_scale):
    """Allow rough approach while far away, then align near the gripper."""
    if distance_scale > 3.0:
        return 40.0, 28.0
    if distance_scale > 1.5:
        return 25.0, 16.0
    return float(TURN_ENTER_DEGREES), float(TURN_EXIT_DEGREES)


def match_target(gems, locked, max_shift=30):
    matches = [gem for gem in gems if gem['colour'] == locked['colour']
               and (gem['x']-locked['x'])**2 + (gem['y']-locked['y'])**2
               <= max_shift**2]
    return min(matches, key=lambda gem:
               (gem['x']-locked['x'])**2 + (gem['y']-locked['y'])**2) if matches else None


def gripper_gem(robot, gems):
    """Return the detected gem nearest the gripper tip, regardless of old lock."""
    if not gems:
        return None
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    return min(gems, key=lambda gem:
               (gem['x']-tip_x)**2 + (gem['y']-tip_y)**2)


def carried_gem_offset(robot, gem):
    """Store a held gem's tip-relative position in marker-scaled axes."""
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    forward = math.hypot(fx, fy)
    marker_side = float(robot['marker_side_px'])
    if forward < 8 or marker_side <= 0:
        return None
    ux, uy = fx/forward, fy/forward
    rx, ry = -uy, ux
    dx, dy = gem['x']-tip_x, gem['y']-tip_y
    return ((dx*ux + dy*uy)/marker_side,
            (dx*rx + dy*ry)/marker_side)


def carried_gem_position(robot, offset):
    """Project a remembered held-gem offset into the current camera frame."""
    if offset is None:
        return None
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    forward = math.hypot(fx, fy)
    marker_side = float(robot['marker_side_px'])
    if forward < 8 or marker_side <= 0:
        return None
    ux, uy = fx/forward, fy/forward
    rx, ry = -uy, ux
    along, lateral = offset
    return {
        'x': tip_x + marker_side*(along*ux + lateral*rx),
        'y': tip_y + marker_side*(along*uy + lateral*ry),
    }


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


def plan_step(robot, gem, captured=False, angle_override=None,
              turn_threshold=TURN_ENTER_DEGREES):
    """Return (command, pulse seconds, distance px, signed angle degrees)."""
    x, y = robot['x'], robot['y']
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    fx, fy = tip_x-x, tip_y-y
    dx, dy = gem['x']-tip_x, gem['y']-tip_y
    forward = math.hypot(fx, fy)
    distance = math.hypot(dx, dy)
    if forward < 8 or not math.isfinite(distance):
        raise ValueError('พิกัดปลาย Gripper ไม่ชัด')
    measured_angle = math.degrees(math.atan2(fx*dy-fy*dx, fx*dx+fy*dy))
    angle = measured_angle if angle_override is None else float(angle_override)
    marker_side = float(robot['marker_side_px'])
    if marker_side <= 0:
        raise ValueError('Invalid ArUco marker size')
    if captured:
        return 'GRIP', 0, distance, angle
    if abs(angle) > turn_threshold:
        duration = _scaled_pulse(
            abs(angle), turn_threshold, 90,
            TURN_MIN_PULSE_SECONDS, TURN_MAX_PULSE_SECONDS)
        return ('R' if angle > 0 else 'L'), duration, distance, angle
    duration = _scaled_pulse(
        distance/marker_side, .7, 5.0,
        FORWARD_MIN_PULSE_SECONDS, FORWARD_MAX_PULSE_SECONDS)
    return 'F', duration, distance, angle


class DirectionController:
    """Smooth visual headings and avoid rapid left/right reversals."""
    def __init__(self):
        self.samples = []
        self.aligning = False
        self.last_turn_angle = None

    def reset(self):
        self.samples.clear()
        self.aligning = False
        self.last_turn_angle = None

    def plan(self, robot, destination, captured=False):
        raw = plan_step(robot, destination, captured=captured)
        if captured:
            return raw
        self.samples.append(raw[3])
        del self.samples[:-3]
        angle = statistics.median(self.samples)
        marker_side = max(1.0, float(robot['marker_side_px']))
        enter_angle, exit_angle = alignment_limits(raw[2]/marker_side)

        crossed_centre = (self.last_turn_angle is not None
                           and angle*self.last_turn_angle < 0
                           and abs(angle) <= enter_angle)
        if crossed_centre or abs(angle) <= exit_angle:
            self.aligning = False
        elif abs(angle) > enter_angle:
            self.aligning = True

        threshold = exit_angle if self.aligning else enter_angle
        result = plan_step(robot, destination, angle_override=angle,
                           turn_threshold=threshold)
        if result[0] in ('L', 'R'):
            self.last_turn_angle = angle
        else:
            self.last_turn_angle = None
        return result


def plan_delivery_step(robot, hole, direction=None, carried_gem=None):
    """Centre the held gem on its colour hole, then request a drop."""
    tip_x, tip_y = robot['gripper_tip_x'], robot['gripper_tip_y']
    carried_x = tip_x if carried_gem is None else carried_gem['x']
    carried_y = tip_y if carried_gem is None else carried_gem['y']
    # Guide the gripper tip to a compensated point so that an off-centre gem
    # between the jaws, rather than the gripper tip itself, reaches the centre.
    destination = {
        'x': hole['x']-(carried_x-tip_x),
        'y': hole['y']-(carried_y-tip_y),
    }
    direction = direction or DirectionController()
    command, duration, distance, angle = direction.plan(robot, destination)
    drop_distance = max(
        DROP_CENTER_MIN_PX,
        min(DROP_CENTER_MAX_PX,
            DROP_CENTER_RADIUS_RATIO*float(hole['radius'])))
    if distance <= drop_distance:
        return 'DROP', 0, distance, angle
    return command, duration, distance, angle


class ApproachRunner:
    def __init__(self, snapshot, send, stop, ready, guide=None,
                 select_colour=None):
        self.snapshot = snapshot
        self.send = send
        self.stop_motors = stop
        self.ready = ready
        self.guide = guide or (lambda colour: None)
        self.select_colour = select_colour or (lambda colour: None)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.running = False
        self.message = 'ยังไม่เริ่ม'
        self.pulses = 0
        self.target = None
        self.carried_offset = None
        self.pickup_colour = AUTO_COLOUR_SEQUENCE[0]

    def state(self):
        with self.lock:
            return dict(running=self.running, message=self.message,
                        pulses=self.pulses, target=self.target,
                        pickup_colour=self.pickup_colour)

    def _message(self, value):
        with self.lock:
            self.message = value

    def start(self):
        with self.lock:
            if self.running:
                raise ValueError('รถกำลังวิ่งไปหาเป้าหมายอยู่')
            if not self.ready():
                raise ValueError('ต้องเปิดกล้องจริง อยู่แท็บควบคุมรถ และปิดการควบคุมด้วยปุ่ม')
            # Every new Auto run starts from GREEN, even if a previous run had
            # already advanced to CYAN.  Changing the vision filter can
            # invalidate its latest observation, so allow the camera worker a
            # short time to publish a fresh GREEN-only frame.
            self.pickup_colour = AUTO_COLOUR_SEQUENCE[0]
            self.select_colour(self.pickup_colour)
            deadline = time.monotonic() + 1.0
            observation = self.snapshot()
            while (self.ready() and time.monotonic() < deadline
                   and (observation is None
                        or observation.get('target') is None
                        or observation['target'].get('colour')
                        != self.pickup_colour)):
                time.sleep(.025)
                observation = self.snapshot()
            if (observation is None or observation['robot'] is None
                    or observation['target'] is None
                    or observation['target']['colour'] != self.pickup_colour
                    or time.monotonic()-observation['time'] > .8):
                raise ValueError(
                    f'ยังไม่พบ ArUco ID 0 คันเดียวและหินสี {self.pickup_colour} ในภาพล่าสุด')
            target_colour = observation['target']['colour']
            if target_colour not in observation.get('holes', {}):
                raise ValueError(f'ยังไม่ได้ตั้งวงรับหินสี {target_colour}')
            self.target = target_copy(observation['target'])
            self.running = True
            self.pulses = 0
            self.message = 'กำลังยืนยันหินเป้าหมาย'
            self.cancel.clear()
            threading.Thread(target=self._run, daemon=True).start()

    def abort(self):
        self.cancel.set()
        self.stop_motors()
        self.guide(None)
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

    def _drive_pulse(self, command, duration):
        """Request one small firmware-timed pulse and return the next-move time."""
        self.pulses += 1
        # Firmware accepts at most 500 ms per timed command.  Split longer
        # movements (such as backing away after a drop) into safe pulses while
        # retaining cancellation and the firmware's automatic stop safeguard.
        remaining = duration
        while remaining > 0:
            chunk = min(MAX_MOTOR_PULSE_SECONDS, remaining)
            milliseconds = max(20, round(chunk*1000))
            if not self._send_continuous(f'P{command},{milliseconds}\n'):
                return None
            if self.cancel.wait(milliseconds/1000):
                self.stop_motors()
                return None
            remaining -= chunk
        # This redundant stop is fail-safe; normal pulse timing is local to ESP32.
        self.stop_motors()
        return time.monotonic() + FEEDBACK_DELAY_SECONDS

    def _show_feedback_delay(self, move_after, command, distance, angle,
                             prefix=''):
        """Keep reporting fresh geometry while waiting to make the next move."""
        remaining = move_after - time.monotonic()
        if remaining <= 0:
            return False
        lead = f'{prefix}: ' if prefix else ''
        self._message(
            f'{lead}คำนวณก่อนขยับ {command} | ห่าง {distance:.0f} px | '
            f'มุม {angle:.0f}° | เหลือ {remaining:.1f} วิ')
        return True

    def _run(self):
        last_frame = -1
        stable = 0
        no_progress = 0
        previous_pose = None
        previous_turn_error = None
        wrong_turns = 0
        capture_frames = 0
        phase = 'pickup'
        next_move_at = 0.0
        direction = DirectionController()
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
                    if (candidate is None
                            or candidate['colour'] != self.pickup_colour):
                        self.stop_motors()
                        self._message(f'รอหินสี {self.pickup_colour} ก้อนถัดไป')
                        self.cancel.wait(.10)
                        continue
                    if candidate['colour'] not in observation.get('holes', {}):
                        self.stop_motors()
                        self._message(f'รอการตั้งวงรับหินสี {candidate["colour"]}')
                        self.cancel.wait(.25)
                        continue
                    self.target = target_copy(candidate)
                    stable = 0
                    capture_frames = 0
                    previous_pose = None
                    previous_turn_error = None
                    wrong_turns = 0
                    no_progress = 0
                    direction.reset()
                    self._message(f'เลือกหินก้อนใหม่สี {self.target["colour"]}')
                if phase == 'deliver':
                    # Once closing was triggered by consecutive live detections,
                    # keep that capture latched until the planned drop.  A held
                    # gem is commonly hidden by the closed jaws, so missing HSV
                    # detections are not evidence that it was dropped.
                    carried = gripper_gem(
                        robot, observation.get('gripper_gems', []))
                    if carried is not None:
                        self.carried_offset = carried_gem_offset(robot, carried)
                    if (carried is not None
                            and carried['colour'] != self.target['colour']):
                        self.target = target_copy(carried)
                        self.guide(self.target['colour'])
                    hole = observation.get('holes', {}).get(self.target['colour'])
                    if hole is None:
                        self.stop_motors()
                        self._message(f'รอวงรับหินสี {self.target["colour"]}')
                        self.cancel.wait(.25)
                        continue
                    carried_position = carried_gem_position(
                        robot, self.carried_offset)
                    command, duration, distance, angle = plan_delivery_step(
                        robot, hole, direction, carried_position)
                    if self._show_feedback_delay(
                            next_move_at, command, distance, angle,
                            f'พาหินไปวง {self.target["colour"]}'):
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
                        if self.cancel.wait(.7):
                            return
                        completed_colour = self.target['colour']
                        self._message(
                            f'วางหินสี {completed_colour} แล้ว; กำลังถอยออกจากวง')
                        next_move_at = self._drive_pulse(
                            'B', BACK_AWAY_PULSE_SECONDS)
                        if next_move_at is None:
                            return
                        self.target = None
                        self.carried_offset = None
                        self.guide(None)
                        if completed_colour == AUTO_COLOUR_SEQUENCE[0]:
                            self.pickup_colour = AUTO_COLOUR_SEQUENCE[1]
                        self.select_colour(self.pickup_colour)
                        phase = 'pickup'
                        stable = 0
                        capture_frames = 0
                        previous_pose = None
                        previous_turn_error = None
                        wrong_turns = 0
                        no_progress = 0
                        direction.reset()
                        self._message(
                            f'วางหินในวงสี {completed_colour} แล้ว; '
                            f'รอหินสี {self.pickup_colour} ก้อนถัดไป')
                        continue
                    self._message(
                        f'พาหินไปวง {self.target["colour"]}: {command} | '
                        f'ห่าง {distance:.0f} px | มุม {angle:.0f}° | '
                        f'pulse {duration*1000:.0f} ms')
                    previous_pose = current_pose
                    previous_turn_error = abs(angle) if command in ('L', 'R') else None
                    next_move_at = self._drive_pulse(command, duration)
                    if next_move_at is None:
                        return
                    continue
                gem = match_target(observation['gems'], self.target)
                if gem is None:
                    # A gem can disappear behind the gripper as the car gets
                    # close. Continue toward the last confirmed coordinates.
                    gem = self.target
                    self._message('มองไม่เห็นเป้าหมาย; ไปยังพิกัดล่าสุดต่อ')
                else:
                    self.target = target_copy(gem)
                # The jaw ROI is stronger evidence than the old target lock:
                # close for any stone actually seen between the gripper arms.
                captured_gem = gripper_gem(
                    robot, observation.get('gripper_gems', []))
                capture_frames = capture_frames + 1 if captured_gem is not None else 0
                if stable < 3:
                    stable += 1
                    self._message(f'ยืนยันเป้าหมาย {stable}/3 เฟรม')
                    continue
                # Require two consecutive camera frames with the target centre
                # inside the jaw ROI before closing the gripper.
                command, duration, distance, angle = direction.plan(
                    robot, captured_gem or gem, captured=capture_frames >= 2)
                if self._show_feedback_delay(
                        next_move_at, command, distance, angle):
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
                    # Deliver according to the stone that is physically in the
                    # gripper, even if it differs from the earlier visual lock.
                    self.target = target_copy(captured_gem)
                    self.carried_offset = carried_gem_offset(
                        robot, captured_gem)
                    self.guide(self.target['colour'])
                    self.stop_motors()
                    self._message(f'พบหินในกริปเปอร์ {distance:.0f} px; กำลังคีบ')
                    if not self._send_continuous('C'):
                        return
                    if self.cancel.wait(.7):
                        return
                    self._message(f'คีบหินแล้ว; กำลังไปวงสี {self.target["colour"]}')
                    phase = 'deliver'
                    direction.reset()
                    previous_pose = None
                    previous_turn_error = None
                    wrong_turns = 0
                    no_progress = 0
                    continue
                self._message(
                    f'{command} สั้น ๆ | ห่าง {distance:.0f} px | '
                    f'มุม {angle:.0f}° | pulse {duration*1000:.0f} ms')
                previous_pose = current_pose
                previous_turn_error = abs(angle) if command in ('L', 'R') else None
                next_move_at = self._drive_pulse(command, duration)
                if next_move_at is None:
                    return
        except (ValueError, ConnectionError) as exc:
            # Unexpected command/geometry errors still leave the motors safe.
            # The outer controller can be started again after the cause is fixed.
            self._message(str(exc))
        finally:
            self.stop_motors()
            self.guide(None)
            with self.lock:
                self.running = False
