import unittest

from auto_vision_gem_camera_v4_1 import (
    AUTO_COLOUR_ORDER, choose_nearest_target, target_route,
)


class TargetRouteTests(unittest.TestCase):
    def setUp(self):
        self.robot = {
            'aruco_id': 0,
            'x': 0, 'y': 0,
            'gripper_tip_x': 10, 'gripper_tip_y': 0,
            'marker_side_px': 10,
        }

    def test_prefers_forward_gem_over_closer_gem_behind_within_colour(self):
        behind = {'colour': 'RED', 'x': -5, 'y': 0}
        forward = {'colour': 'RED', 'x': 35, 'y': 0}

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

    def test_still_prefers_nearest_when_turns_are_equal_within_colour(self):
        near = {'colour': 'RED', 'x': 30, 'y': 0}
        far = {'colour': 'RED', 'x': 50, 'y': 0}

        selected = choose_nearest_target([self.robot], [far, near])

        self.assertEqual(selected[1], near)

    def test_uses_requested_colour_order_before_route_cost(self):
        gems = [
            {'colour': 'BLUE', 'x': 12, 'y': 0},
            {'colour': 'ORANGE', 'x': 14, 'y': 0},
            {'colour': 'RED', 'x': 16, 'y': 0},
            {'colour': 'PURPLE', 'x': 18, 'y': 0},
            {'colour': 'GREEN', 'x': 20, 'y': 0},
            {'colour': 'CYAN', 'x': 100, 'y': 0},
        ]

        selected = choose_nearest_target([self.robot], gems)

        self.assertEqual(AUTO_COLOUR_ORDER,
                         ('CYAN', 'GREEN', 'PURPLE', 'RED', 'ORANGE', 'BLUE'))
        self.assertEqual(selected[1]['colour'], 'CYAN')

    def test_advances_when_earlier_colours_are_gone(self):
        gems = [
            {'colour': 'BLUE', 'x': 12, 'y': 0},
            {'colour': 'RED', 'x': 30, 'y': 0},
            {'colour': 'PURPLE', 'x': 80, 'y': 0},
        ]

        selected = choose_nearest_target([self.robot], gems)

        self.assertEqual(selected[1]['colour'], 'PURPLE')


if __name__ == '__main__':
    unittest.main()
