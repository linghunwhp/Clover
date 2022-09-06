import numpy as np
import math


def gs_kernel(dist, h):
    m = np.shape(dist)[0]
    one = 1 / (h * math.sqrt(2 * math.pi))
    two = np.mat(np.zeros((m, 1)))
    for i in range(m):
        two[i, 0] = (-0.5 * dist[i] * dist[i].T) / (h * h)
        two[i, 0] = np.exp(two[i, 0])

    gs_val = one * two
    return gs_val


def shift_point(point, points, h):
    points = np.mat(points)
    m = np.shape(points)[0]
    point_dist = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_dist[i, 0] = cosine_similarity(point, points[i])

    point_weights = gs_kernel(point_dist, h)
    all_sum = 0.0
    for i in range(m):
        all_sum += point_weights[i, 0]

    point_shifted = point_weights.T * points / all_sum
    return point_shifted


def lb_points(mean_shift_points):
    lb_list = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        lb_list.append(index_dict[item_1])
    return lb_list


def cosine_similarity(pointA, pointB):
    return np.dot(pointA.flatten(), pointB.flatten())/(np.linalg.norm(pointA.flatten())*np.linalg.norm(pointB.flatten()))


def mean_shift(points, h=2, MIN_DISTANCE=0.001):
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0
    m = np.shape(mean_shift_points)[0]
    need_shift = [True] * m

    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print("iteration : " + str(iteration))
        for i in range(0, m):
            if not need_shift[i]:
                continue
            point_new = mean_shift_points[i]
            point_new_start = point_new
            point_new = shift_point(point_new, points, h)
            dist = cosine_similarity(point_new, point_new_start)

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:
                need_shift[i] = False

            mean_shift_points[i] = point_new
    lb = lb_points(mean_shift_points)
    return lb
