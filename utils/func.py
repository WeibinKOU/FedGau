import os

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean


def get_dir(*path):
    if len(path) == 0:
        dir_path = path[0]
    else:
        dir_path = os.path.join(*path)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def cal_keypoints(mask):
    for threshold in range(240, 0, -16):
        if threshold <= 96:
            return None

        tmp_mask = (mask >= threshold).astype(np.uint8)
        n, components = cv2.connectedComponents(tmp_mask)
        if n >= 5:
            break

    points = []
    v = pd.value_counts(components.ravel())
    for i, t in v.iloc[1:5].items():
        xs, ys = np.where(components==i)
        x, y = xs.mean(), ys.mean()
        points.append((x, y))

    points = np.array(points)
    if points.shape != (4, 2):
        return None

    dists_triangle = np.sum(distance_matrix(points, points), axis=1)
    purlicue_index = np.argmax(dists_triangle)
    purlicue_point = points[purlicue_index]
    second_index = np.argmin(dists_triangle)
    second_point = points[second_index]
    other_points = np.array([points[x] for x in range(4) if x!=purlicue_index and x!=second_index])

    d1 = euclidean(purlicue_point, other_points[0]) + euclidean(second_point, other_points[0])
    d2 = euclidean(purlicue_point, other_points[1]) + euclidean(second_point, other_points[1])
    if d1 > d2:
        first_point, third_point = other_points
    else:
        first_point, third_point = other_points[1], other_points[0]

    return np.array([first_point, second_point, third_point, purlicue_point])
