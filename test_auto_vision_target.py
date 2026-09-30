import unittest

from auto_vision_gem_camera_v4_1 import choose_nearest_target, target_route


class TargetRouteTests(unittest.TestCase):
    def setUp(self):
        self.robot = {
            'aruco_id': 0,
            'x': 0, 'y': 0,
            'gripper_tip_x': 10, 'gripper_tip_y': 0,
            'marker_side_px': 10,
        }

    def test_prefers_forward_gem_over_closer_gem_behind(self):
        behind = {'colour': 'RED', 'x': -5, 'y': 0}
        forward = {'colour': 'BLUE', 'x': 35, 'y': 0}

        selected = choose_nearest_target([self.robot], [behind, forward])

        self.assertEqual(selected[1], forward)

    def test_route_uses_shortest_signed_turn_direction(self):
        _, left_angle, _ = target_route(
            self.robot, {'colour': 'RED', 'x': 10, 'y': -20})
        _, right_angle, _ = target_route(
            self.robot, {'colour': 'BLUE', 'x': 10, 'y': 20})

        self.assertAlmostEqual(abs(left_angle), 90)
        self.assertAlmostEqual(abs(right_angle), 90)
        self.assertLess(left_angle, 0)
        self.assertGreater(right_angle, 0)

    def test_still_prefers_nearest_when_turns_are_equal(self):
        near = {'colour': 'RED', 'x': 30, 'y': 0}
        far = {'colour': 'BLUE', 'x': 50, 'y': 0}

        selected = choose_nearest_target([self.robot], [far, near])

        self.assertEqual(selected[1], near)


if __name__ == '__main__':
    unittest.main()
