import json
import unittest

import cv2
import numpy as np

from auto_vision_gem_colors_v4_1 import detect


class GemCandidateShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('hsv_colour_config.json', encoding='utf-8') as stream:
            cls.config = json.load(stream)

    def cyan_bgr(self):
        item = self.config['colours']['CYAN']
        hsv = np.uint8([[[item['h_center'], item['s_center'], item['v_center']]]])
        return tuple(int(value) for value in
                     cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def green_bgr(self):
        item = self.config['colours']['GREEN']
        hsv = np.uint8([[[item['h_center'], item['s_center'], item['v_center']]]])
        return tuple(int(value) for value in
                     cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def test_thin_cyan_stroke_is_not_a_gem(self):
        frame = np.zeros((220, 1200, 3), np.uint8)
        cv2.line(frame, (100, 80), (260, 80), self.cyan_bgr(), 3)

        _, gems, _, _ = detect(frame, hsv_config=self.config)

        self.assertFalse(any(gem['colour'] == 'CYAN' for gem in gems))

    def test_compact_cyan_blob_is_still_a_gem(self):
        frame = np.zeros((220, 1200, 3), np.uint8)
        cv2.circle(frame, (180, 100), 9, self.cyan_bgr(), -1)

        _, gems, _, _ = detect(frame, hsv_config=self.config)

        cyan = [gem for gem in gems if gem['colour'] == 'CYAN']
        self.assertEqual(len(cyan), 1)
        self.assertAlmostEqual(cyan[0]['x'], 180, delta=1)
        self.assertAlmostEqual(cyan[0]['y'], 100, delta=1)

    def test_active_colours_returns_only_green_detections(self):
        frame = np.zeros((220, 1200, 3), np.uint8)
        cv2.circle(frame, (180, 100), 9, self.cyan_bgr(), -1)
        cv2.circle(frame, (300, 100), 9, self.green_bgr(), -1)

        _, gems, masks, _ = detect(
            frame, hsv_config=self.config, active_colours=('GREEN',))

        self.assertEqual(tuple(masks), ('GREEN',))
        self.assertEqual([gem['colour'] for gem in gems], ['GREEN'])


if __name__ == '__main__':
    unittest.main()
