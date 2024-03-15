import unittest
import torch

from iou import intersection_over_union


class TestIntersectionOverUnion(unittest.TestCase):
    def setUp(self):
        # testcases we want to run
        self.t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
        self.t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
        self.t1_correct_iou = 1 / 7

        self.t2_box1 = torch.tensor([0.95, 0.6, 0.5, 0.2])
        self.t2_box2 = torch.tensor([0.95, 0.7, 0.3, 0.2])
        self.t2_correct_iou = 3 / 13

        self.t3_box1 = torch.tensor([0.25, 0.15, 0.3, 0.1])
        self.t3_box2 = torch.tensor([0.25, 0.35, 0.3, 0.1])
        self.t3_correct_iou = 0

        self.t4_box1 = torch.tensor([0.7, 0.95, 0.6, 0.1])
        self.t4_box2 = torch.tensor([0.5, 1.15, 0.4, 0.7])
        self.t4_correct_iou = 3 / 31

        # accept if the different in iou is small
        self.epsilon = 0.001

    def test_both_inside_cell_shares_area(self):
        iou = intersection_over_union(self.t1_box1, self.t1_box2, box_format='midpoint')
        self.assertTrue((torch.abs(iou - self.t1_correct_iou) < self.epsilon))

    def test_partially_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t2_box1, self.t2_box2, box_format='midpoint')
        self.assertTrue((torch.abs(iou - self.t2_correct_iou) < self.epsilon))

    def test_both_inside_cell_shares_no_area(self):
        iou = intersection_over_union(self.t3_box1, self.t3_box1, box_format='midpoint')
        self.assertTrue((torch.abs(iou - self.t3_correct_iou) < self.epsilon))

    def test_midpoint_outside_shares_area(self):
        iou = intersection_over_union(self.t4_box1, self.t4_box2, box_format='midpoint')
        self.assertTrue((torch.abs(iou - self.t4_correct_iou) < self.epsilon))


if __name__ == '__main__':
    print('Running Intersection Over Union Test...')
    unittest.main()
