import torch
import math


def get_day_night_weights(phi, steepness=1.0, min_w=0, max_w=1):
    w_day_raw = (-torch.cos(phi) + 1.0) / 2.0

    if steepness != 1.0:
        w_day_raw = torch.pow(w_day_raw, steepness)

    w_day = w_day_raw * (max_w - min_w) + min_w

    return w_day, 1.0 - w_day