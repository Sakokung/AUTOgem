import unittest
import time

from auto_approach import (
    ApproachRunner, gem_in_gripper, match_target, plan_delivery_step, plan_step,
)


class AutoApproachTests(unittest.TestCase):
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
            # Continuous mode would start looking for another stone after the
            # final open command. Stop explicitly so this unit test can finish.
            if packets == ['O', 'C', 'O']:
                runner.cancel.set()

        runner = ApproachRunner(snapshot, send, lambda: None, lambda: True)
        runner.target = dict(gem)
        runner.running = True
        runner._run()

        self.assertEqual(packets, ['O', 'C', 'O'])
        self.assertIn('วางหินในวงสี RED แล้ว', runner.state()['message'])


if __name__ == '__main__':
    unittest.main()
