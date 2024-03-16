import torch
import unittest

from nms import non_max_suppression


class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        # test case we want to run
        self.t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5], # box 1
            [1, 0.8, 0.5, 0.5, 0.2, 0.4], # box 2
            [1, 0.7, 0.25, 0.35, 0.3, 0.1], # box 3
            [1, 0.05, 0.1, 0.1, 0.1, 0.1] # box 4
        ]
        self.c1_boxes = [
            [1, 1, 0.45, 0.4, 0.5],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1]
        ]

        self.t2_boxes = [
            [1, 1, 0.45, 0.4, 0.5], # box 1
            [2, 0.9, 0.5, 0.5, 0.2, 0.4], # box 2
            [1, 0.8, 0.25, 0.35, 0.3, 0.1], # box 3
            [1, 0.05, 0.1, 0.1, 0.1, 0.1], # box 4
        ]
        self.c2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1]
        ]

        self.t3_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5], # box 1
            [1, 1, 0.5, 0.5, 0.2, 0.4], # box 2
            [2, 0.8, 0.25, 0.35, 0.3, 0.1], # box 3
            [1, 0.05, 0.1, 0.1, 0.1, 0.1] # box 4
        ]
        self.c3_boxes = [
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1]
        ]

        self.t4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5], # box 1
            [1, 1, 0.5, 0.5, 0.2, 0.4], # box 2
            [1, 0.8, 0.25, 0.35, 0.3, 0.1], # box 3
            [1, 0.05, 0.1, 0.1, 0.1, 0.1] # box 4
        ]
        self.c4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1]
        ]

    def remove_on_iou(self):
        bboxes = non_max_suppression(
            self.t1_boxes,
            prob_threshold=0.2,
            iou_threshold=7/20,
            box_format='midpoint'
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = non_max_suppression(
            self.t2_boxes,
            prob_threshold=0.2,
            iou_threshold=7/20,
            box_format='midpoint'
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = non_max_suppression(
            self.t3_boxes,
            prob_threshold=0.2,
            iou_threshold=7/20,
            box_format='midpoint'
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = non_max_suppression(
            self.t4_boxes,
            prob_threshold=0.2,
            iou_threshold=9/20,
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))


if __name__ == '__main__':
    print('Running Non Max Suppression Test...')
    unittest.main()
