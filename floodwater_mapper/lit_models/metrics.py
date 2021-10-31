import numpy as np


def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.
    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torch.Tensor): a tensor of labels
    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255) # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()