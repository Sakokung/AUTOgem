import unittest

from auto_vision_gem_camera_v4_1 import (
    AUTO_COLOUR_ORDER, choose_nearest_target, target_route,
)
from vision_panel import occupied_hole_colours


class TargetRouteTests(unittest.TestCase):
    def setUp(self):
        self.robot = {
            'aruco_id': 0,
            'x': 0, 'y': 0,
            'gripper_tip_x': 10, 'gripper_tip_y': 0,
            'marker_side_px': 10,
        }

    def test_prefers_forward_gem_over_closer_gem_behind_within_colour(self):
        behind = {'colour': 'GREEN', 'x': -5, 'y': 0}
        forward = {'colour': 'GREEN', 'x': 35, 'y': 0}

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
        near = {'colour': 'GREEN', 'x': 30, 'y': 0}
        far = {'colour': 'GREEN', 'x': 50, 'y': 0}

        selected = choose_nearest_target([self.robot], [far, near])

        self.assertEqual(selected[1], near)

    def test_auto_mode_selects_only_green(self):
        gems = [
            {'colour': 'BLUE', 'x': 12, 'y': 0},
            {'colour': 'ORANGE', 'x': 14, 'y': 0},
            {'colour': 'RED', 'x': 16, 'y': 0},
            {'colour': 'PURPLE', 'x': 18, 'y': 0},
            {'colour': 'GREEN', 'x': 20, 'y': 0},
            {'colour': 'CYAN', 'x': 100, 'y': 0},
        ]

        selected = choose_nearest_target([self.robot], gems)

        self.assertEqual(AUTO_COLOUR_ORDER, ('GREEN', 'CYAN'))
        self.assertEqual(selected[1]['colour'], 'GREEN')

    def test_green_is_selected_even_when_other_colours_are_closer(self):
        gems = [
            {'colour': 'BLUE', 'x': 12, 'y': 0},
            {'colour': 'RED', 'x': 14, 'y': 0},
            {'colour': 'PURPLE', 'x': 16, 'y': 0},
            {'colour': 'GREEN', 'x': 100, 'y': 0},
        ]

        selected = choose_nearest_target([self.robot], gems)

        self.assertEqual(selected[1]['colour'], 'GREEN')

    def test_cyan_is_next_when_green_is_gone(self):
        gems = [
            {'colour': 'BLUE', 'x': 12, 'y': 0},
            {'colour': 'RED', 'x': 30, 'y': 0},
            {'colour': 'PURPLE', 'x': 80, 'y': 0},
            {'colour': 'CYAN', 'x': 100, 'y': 0},
        ]

        selected = choose_nearest_target([self.robot], gems)

        self.assertEqual(selected[1]['colour'], 'CYAN')

    def test_explicit_green_filter_does_not_advance_early(self):
        gems = [
            {'colour': 'CYAN', 'x': 12, 'y': 0},
            {'colour': 'GREEN', 'x': 100, 'y': 0},
        ]

        selected = choose_nearest_target(
            [self.robot], gems, colour_order=('GREEN',))

        self.assertEqual(selected[1]['colour'], 'GREEN')

    def test_colour_is_blocked_when_matching_stone_is_inside_its_circle(self):
        gems = [
            {'colour': 'RED', 'x': 108, 'y': 100},
            {'colour': 'RED', 'x': 20, 'y': 20},
            {'colour': 'BLUE', 'x': 200, 'y': 200},
        ]
        holes = {
            'RED': {'x': 100, 'y': 100, 'radius': 10},
            'BLUE': {'x': 300, 'y': 300, 'radius': 10},
        }

        occupied = occupied_hole_colours(gems, holes)
        available = [gem for gem in gems if gem['colour'] not in occupied]

        self.assertEqual(occupied, {'RED'})
        self.assertEqual([gem['colour'] for gem in available], ['BLUE'])

    def test_other_colour_inside_circle_does_not_block_that_circle_colour(self):
        gems = [{'colour': 'BLUE', 'x': 100, 'y': 100}]
        holes = {'RED': {'x': 100, 'y': 100, 'radius': 10}}

        self.assertEqual(occupied_hole_colours(gems, holes), set())


if __name__ == '__main__':
    unittest.main()
