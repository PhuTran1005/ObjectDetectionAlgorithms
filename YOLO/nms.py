import torch

from iou import intersection_over_union


def non_max_suppression(
    bboxes,
    iou_threshold,
    prob_threshold,
    box_format='corners'
):
    """Non Max Suppresion implementation to clean bboxes

    Args:
        bboxes (list): list of list contain all bboxes with each bboxes. Specifically, [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        prob_threshold (float): threshold to remove predicted bboxes (not good bboxes)
        box_format (str, optional): 'midpoint' or 'corners' used to specify bboxes. Defaults to 'corners'.

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(boxes) == list

    # remove bboxes which have the prob < prob_threshold
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = list()

    while bboxes:
        chosen_box = bboxes.pop(0) # get the box has a highest prob_score

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format)
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
