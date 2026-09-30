import unittest
import time

from auto_approach import (
    FEEDBACK_DELAY_SECONDS, ApproachRunner, DirectionController, gem_in_gripper,
    gripper_gem, match_target, plan_delivery_step, plan_step,
)


class AutoApproachTests(unittest.TestCase):
    def test_drive_pulse_returns_immediately_with_1_5_second_feedback_deadline(self):
        packets = []
        runner = ApproachRunner(
            lambda: None, packets.append, lambda: None, lambda: True)

        started = time.monotonic()
        move_after = runner._drive_pulse('F', .02)
        elapsed = time.monotonic() - started

        self.assertEqual(FEEDBACK_DELAY_SECONDS, 1.5)
        self.assertEqual(packets, ['PF,20\n'])
        self.assertLess(elapsed, .5)
        self.assertAlmostEqual(
            move_after - time.monotonic(), FEEDBACK_DELAY_SECONDS, delta=.1)

    def test_feedback_delay_displays_fresh_distance_and_angle(self):
        runner = ApproachRunner(
            lambda: None, lambda packet: None, lambda: None, lambda: True)

        waiting = runner._show_feedback_delay(
            time.monotonic() + 1.5, 'R', 123.4, -17.6)

        self.assertTrue(waiting)
        self.assertIn('ห่าง 123 px', runner.state()['message'])
        self.assertIn('มุม -18°', runner.state()['message'])

    def test_locked_target_remains_usable_when_detection_is_missing(self):
        locked = {'colour': 'RED', 'x': 100, 'y': 50}
        self.assertIsNone(match_target([], locked))

        robot = {
            'x': 20, 'y': 50,
            'gripper_tip_x': 40, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        command, _, distance, angle = plan_step(robot, locked)
        self.assertEqual(command, 'F')
        self.assertEqual(distance, 60)
        self.assertEqual(angle, 0)

    def test_gem_inside_gripper_requests_grip(self):
        robot = {
            'x': 20, 'y': 50,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        gem = {'colour': 'RED', 'x': 82, 'y': 55}
        self.assertTrue(gem_in_gripper(robot, gem))
        self.assertNotEqual(plan_step(robot, gem)[0], 'GRIP')
        self.assertEqual(plan_step(robot, gem, captured=True)[0], 'GRIP')

    def test_gem_beside_gripper_is_not_gripped(self):
        robot = {
            'x': 20, 'y': 50,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        gem = {'colour': 'RED', 'x': 80, 'y': 70}
        self.assertFalse(gem_in_gripper(robot, gem))
        self.assertNotEqual(plan_step(robot, gem)[0], 'GRIP')

    def test_any_detected_gripper_gem_is_selected(self):
        robot = {'gripper_tip_x': 80, 'gripper_tip_y': 50}
        old_target = {'colour': 'RED', 'x': 10, 'y': 10}
        inside = {'colour': 'BLUE', 'x': 82, 'y': 51}
        self.assertIsNone(match_target([inside], old_target))
        self.assertEqual(gripper_gem(robot, [inside]), inside)

    def test_delivery_drops_only_when_tip_is_inside_hole(self):
        robot = {
            'x': 20, 'y': 50,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        self.assertEqual(plan_delivery_step(
            robot, {'x': 87, 'y': 50, 'radius': 20})[0], 'DROP')
        self.assertEqual(plan_delivery_step(
            robot, {'x': 150, 'y': 50, 'radius': 20})[0], 'F')

    def test_delivery_prioritizes_distance_for_far_angled_hole(self):
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        command, _, distance, angle = plan_delivery_step(
            robot, {'x': 120, 'y': 57.74, 'radius': 10})
        self.assertGreater(distance/robot['marker_side_px'], 3)
        self.assertAlmostEqual(angle, 30, places=1)
        self.assertEqual(command, 'F')

    def test_delivery_aligns_only_when_close_to_hole(self):
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        command, _, distance, angle = plan_delivery_step(
            robot, {'x': 40, 'y': 11.55, 'radius': 5})
        self.assertLess(distance/robot['marker_side_px'], 1.5)
        self.assertAlmostEqual(angle, 30, places=1)
        self.assertIn(command, ('L', 'R'))

    def test_forward_pulse_gets_shorter_near_target(self):
        robot = {
            'x': 0, 'y': 50,
            'gripper_tip_x': 20, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        near = plan_step(robot, {'x': 40, 'y': 50})
        far = plan_step(robot, {'x': 140, 'y': 50})
        self.assertEqual(near[0], 'F')
        self.assertEqual(far[0], 'F')
        self.assertLess(near[1], far[1])
        self.assertGreaterEqual(near[1], .03)
        self.assertLessEqual(far[1], .10)

    def test_turn_pulse_gets_shorter_for_small_angle(self):
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        slight = plan_step(robot, {'x': 120, 'y': 30})
        sharp = plan_step(robot, {'x': 20, 'y': 100})
        self.assertIn(slight[0], ('L', 'R'))
        self.assertIn(sharp[0], ('L', 'R'))
        self.assertLess(slight[1], sharp[1])
        self.assertGreaterEqual(slight[1], .025)
        self.assertLessEqual(sharp[1], .055)

    def test_small_six_degree_error_is_inside_straight_hysteresis(self):
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        command, duration, _, angle = plan_step(
            robot, {'x': 120, 'y': 10.51})
        self.assertGreater(abs(angle), 5)
        self.assertEqual(command, 'F')
        self.assertLessEqual(duration, .10)

    def test_roughly_aligned_ten_degree_target_moves_forward(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        result = controller.plan(robot, {'x': 120, 'y': 17.63})
        self.assertAlmostEqual(result[3], 10, places=1)
        self.assertEqual(result[0], 'F')

    def test_clearly_misaligned_target_still_turns_first(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        result = controller.plan(robot, {'x': 120, 'y': 100})
        self.assertGreater(abs(result[3]), 40)
        self.assertIn(result[0], ('L', 'R'))

    def test_far_target_prioritizes_reducing_distance(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        result = controller.plan(robot, {'x': 120, 'y': 57.74})
        self.assertAlmostEqual(result[3], 30, places=1)
        self.assertEqual(result[0], 'F')

    def test_near_target_with_same_angle_aligns_before_gripping(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        result = controller.plan(robot, {'x': 40, 'y': 11.55})
        self.assertAlmostEqual(result[3], 30, places=1)
        self.assertIn(result[0], ('L', 'R'))

    def test_direction_filter_ignores_one_opposite_noisy_frame(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        right = {'x': 120, 'y': 100}
        noisy_left = {'x': 120, 'y': -100}

        self.assertEqual(controller.plan(robot, right)[0], 'R')
        self.assertEqual(controller.plan(robot, right)[0], 'R')
        self.assertEqual(controller.plan(robot, noisy_left)[0], 'R')

    def test_small_centre_crossing_does_not_reverse_the_turn(self):
        controller = DirectionController()
        robot = {
            'x': 0, 'y': 0,
            'gripper_tip_x': 20, 'gripper_tip_y': 0,
            'marker_side_px': 20,
        }
        right = {'x': 120, 'y': 100}
        small_left = {'x': 120, 'y': -10.51}

        controller.plan(robot, right)
        controller.plan(robot, right)
        controller.plan(robot, small_left)
        result = controller.plan(robot, small_left)

        self.assertEqual(result[0], 'F')

    def test_auto_opens_then_closes_when_gem_is_in_gripper(self):
        robot = {
            'x': 20, 'y': 50, 'heading_deg': 0,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        gem = {'colour': 'RED', 'x': 82, 'y': 55}
        frame_index = 0
        packets = []

        def snapshot():
            nonlocal frame_index
            frame_index += 1
            return {
                'time': time.monotonic(), 'frame_index': frame_index,
                'robot': robot, 'gems': [gem], 'gripper_gems': [gem],
                'holes': {'RED': {'x': 82, 'y': 55, 'radius': 20}},
                'target': gem,
            }

        runner = None

        def send(packet):
            packets.append(packet)

        runner = ApproachRunner(snapshot, send, lambda: None, lambda: True)
        def back_away(command, duration):
            packets.append(f'P{command},{round(duration*1000)}\n')
            runner.cancel.set()
            return time.monotonic()
        runner._drive_pulse = back_away
        runner.target = dict(gem)
        runner.running = True
        runner._run()

        self.assertEqual(packets, ['O', 'C', 'O', 'PB,60\n'])
        self.assertIn('วางหินในวงสี RED แล้ว', runner.state()['message'])

    def test_auto_closes_for_a_different_gem_seen_inside_gripper(self):
        robot = {
            'x': 20, 'y': 50, 'heading_deg': 0,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        locked = {'colour': 'RED', 'x': 20, 'y': 20}
        captured = {'colour': 'BLUE', 'x': 82, 'y': 51}
        frame_index = 0
        packets = []
        guides = []
        runner = None

        def snapshot():
            nonlocal frame_index
            frame_index += 1
            return {
                'time': time.monotonic(), 'frame_index': frame_index,
                'robot': robot, 'gems': [locked, captured],
                'gripper_gems': [captured],
                'holes': {
                    'RED': {'x': 20, 'y': 20, 'radius': 20},
                    'BLUE': {'x': 82, 'y': 51, 'radius': 20},
                },
                'target': locked,
            }

        def send(packet):
            packets.append(packet)

        runner = ApproachRunner(snapshot, send, lambda: None, lambda: True,
                                guides.append)
        def back_away(command, duration):
            packets.append(f'P{command},{round(duration*1000)}\n')
            runner.cancel.set()
            return time.monotonic()
        runner._drive_pulse = back_away
        runner.target = dict(locked)
        runner.running = True
        runner._run()

        self.assertEqual(packets, ['O', 'C', 'O', 'PB,60\n'])
        self.assertIn('BLUE', guides)
        self.assertEqual(guides[-1], None)
        self.assertIn('วงสี BLUE', runner.state()['message'])

    def test_auto_keeps_gripper_closed_when_carried_gem_is_occluded(self):
        robot = {
            'x': 20, 'y': 50, 'heading_deg': 0,
            'gripper_tip_x': 80, 'gripper_tip_y': 50,
            'marker_side_px': 20,
        }
        gem = {'colour': 'RED', 'x': 82, 'y': 51}
        frame_index = 0
        packets = []
        guides = []
        runner = None

        def snapshot():
            nonlocal frame_index
            frame_index += 1
            carrying = frame_index <= 4
            return {
                'time': time.monotonic(), 'frame_index': frame_index,
                'robot': robot, 'gems': [gem] if carrying else [],
                'gripper_gems': [gem] if carrying else [],
                'holes': {'RED': {'x': 82, 'y': 51, 'radius': 20}},
                'target': gem if carrying else None,
            }

        def send(packet):
            packets.append(packet)
            if packets == ['O', 'C', 'O', 'PB,60\n']:
                runner.cancel.set()

        runner = ApproachRunner(snapshot, send, lambda: None, lambda: True,
                                guides.append)
        runner.target = dict(gem)
        runner.running = True
        runner._run()

        self.assertEqual(packets, ['O', 'C', 'O', 'PB,60\n'])
        self.assertEqual(packets.count('O'), 2)
        self.assertIn('RED', guides)
        self.assertEqual(guides[-1], None)


if __name__ == '__main__':
    unittest.main()
