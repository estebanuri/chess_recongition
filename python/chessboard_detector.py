import glob
import os

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

import utils

green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
magenta = (255, 255, 0)
yellow = (0, 255, 255)

def create_patchs_synthetic(radius=4):

    size = (radius, radius)
    zeros = np.zeros(size)
    ones = np.ones(size)

    oz = np.hstack((ones, zeros))
    zo = np.hstack((zeros, ones))
    np.vstack((oz, zo))
    patch = np.vstack((oz, zo))

    patch = 255 * patch.astype(dtype='uint8')

    return 255 - patch, patch

def thresold(img):

    block = 255
    C = 3
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    #gray = cv2.blur(gray, ksize=(3, 3))
    # ret, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    #th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)
    return th

def normalize(float_img):
    return (255 * np.round(float_img / np.max(float_img))).astype(dtype='uint8')

def skeletonize(img):

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def cluster(points):

    std = np.std(points)

    ovl_thresh = max(0.1 * std, 3)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=ovl_thresh)
    res = clusterer.fit(points)
    clusters = [[] for _ in range(res.n_clusters_)]
    for pt, label in zip(points, res.labels_):
        clusters[label].append(pt)

    centers = []
    for cluster in clusters:
        center = np.mean(cluster, axis=0)
        centers.append(center)

    return centers

def non_maxima_supression(image):

    max = np.max(image)
    _, image = cv2.threshold(image, 0.9 * max, max, cv2.THRESH_BINARY)
    #_, image = cv2.threshold(image, 0.6 * max, max, cv2.THRESH_BINARY)

    # method 1: using contrib thining method
    #image = normalize(image)
    #thinned = cv2.ximgproc.thinning(image)

    # method 2: using skeletonize
    norm = normalize(image)
    thinned = skeletonize(norm)

    return thinned

def draw_points(image, points, colors=(0, 0, 255), ret=None, rad=8, thick=2):

    if ret is None:
        ret = image.copy()

    if isinstance(colors, tuple):
        use_colors = [colors] * len(points)
    else:
        use_colors = colors

    for pos, color in zip(points, use_colors):
        cv2.circle(ret, utils.pix(pos), rad, color, thick)


    return ret

def improve_subpix(th, points, debug=None):

    if debug:
        s = 1
        w, h = th.shape[1], th.shape[0]
        vis = cv2.resize(th, (s * w, s * h))
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        draw_points(vis, s * points, colors=(0, 0, 255), ret=vis, rad=1, thick=1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    corners = np.float32(points).reshape(-1, 1, 2)
    corners = cv2.cornerSubPix(th, corners, (5, 5), (-1, -1), criteria)
    points = corners.reshape(-1, 2)

    if dbg(debug, 'improve_subpix'):
        draw_points(vis, s * points, colors=(0, 255, 0), ret=vis, rad=1, thick=1)
        utils.imshow('subpix', vis)
        cv2.waitKey()

    return points

def template_match(scene, object, debug=None):

    th = thresold(scene)

    result = cv2.matchTemplate(th, object, cv2.TM_SQDIFF)
    result = non_maxima_supression(result)
    points = np.argwhere(result > 0).astype(float)
    # converts points from (row, column) to (x, y) pairs
    points = points[:, ::-1]

    # add object centering
    obj_w, obj_h = object.shape[1], object.shape[0]
    center = np.array((obj_w, obj_h)) / 2
    points += center

    points = cluster(points)

    # improve points accuracy
    #improve_subpix(th, points, debug)

    return points

def order_points(pts):

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def create_board_layout(wb=True, bw=True, internal=True, board_size=8):

    vertices = []
    # wb_vertices = []
    # bw_vertices = []

    if internal:
        start = 1
        end = board_size
    else:
        start = 0
        end = board_size + 1

    for i in range(start, end):
        for j in range(start, end):

            vertex = (i, j)
            if wb and bw:
                vertices.append(vertex)
            elif wb and (i + j) % 2 == 0:
                vertices.append(vertex)
            elif bw and (i + j) % 2 == 1:
                vertices.append(vertex)

    return vertices
    #return vertices, wb_vertices, bw_vertices

def improve_match(scene_img, nn1, ret, scene_points):

    best_H = ret['H']

    # start from the layout corners
    board_xt_corners = [[1, 1], [1, 7], [7, 7], [7, 1]]
    board_xt_corners = np.float32(board_xt_corners).reshape(-1, 1, 2)
    # transform to scene domain
    scene_xt_corners = cv2.perspectiveTransform(board_xt_corners, best_H)
    scene_xt_corners = scene_xt_corners.reshape(-1, 2)
    # search nearest vertices
    distances, indices = nn1.kneighbors(scene_xt_corners)
    # get candidates
    scene_xt_candidates = scene_points[indices]
    scene_xt_candidates = np.float32(scene_xt_candidates)

    best_H_inv = np.linalg.inv(best_H)
    board_back_corners = cv2.perspectiveTransform(scene_xt_candidates, best_H_inv)

    board_back_corners = np.round(board_back_corners)
    H = cv2.getPerspectiveTransform(board_back_corners, scene_xt_candidates)
    ret['H'] = H

    #utils.imshow('board', draw_points(scene_img, scene_xt_candidates.reshape(-1, 2)))


def better_score(this, other):
    return this > other
    #return this < other

def dbg(debug, key):
    if debug is None:
        return False
    if not key in debug:
        return False
    return debug[key]

def match_layout(scene_img,
                 board_layout,
                 test_layouts,
                 nn1,
                 scene_candidates,
                 best=None,
                 debug=None):

    # nn1 = NearestNeighbors(n_neighbors=1)
    # nn1.fit(scene_points)

    if best is None:
        best = {}
        best['score'] = None

    for test_layout in test_layouts:
        #print("tl:", test_layout)
        test_layout = np.float32(test_layout).reshape(-1, 1, 2)

        H = cv2.getPerspectiveTransform(test_layout, scene_candidates.reshape(-1, 1, 2))
        scene_board_points_candidates = cv2.perspectiveTransform(board_layout, H).reshape(-1, 2)

        distances, indices = nn1.kneighbors(scene_board_points_candidates)
        # np.set_printoptions(suppress=True)
        # print(distances.T)
        #score = np.linalg.norm(distances)
        score = sum(map(lambda d: d < 10, distances))[0]
        #print("current score:", score)
        if best['score'] is None or better_score(score, best['score']):
            #print("better score", score)
            best['score'] = score
            best['H'] = H
            best['layout'] = test_layout

        if dbg(debug, 'match_layout'):
            utils.imshow('board', draw_points(scene_img, scene_board_points_candidates))
            cv2.waitKey()


    return best

def find_bw_quatern(candidates):

    c = candidates[0]
    used = set()
    pairs = []
    for i in range(1, len(candidates)):
        if len(pairs) >= 2:
            break
        if i in used:
            continue
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            b = candidates[j]
            u = b - a
            v = c - a
            d = np.linalg.norm(np.cross(u, v)) / np.linalg.norm(u)

            if d < 1:
                used.add(i)
                used.add(j)
                pairs.append((i, j))
                break

    return pairs


# def choose_test_indices(points, max_indices=4):
#
#     # choose a random point from wb
#     # chosen_idx = random.randint(0, len(points_wb))
#
#     # looks for center-ish points
#     np_points = np.array(points)
#     center = np_points.mean(axis=0)
#     distances = np.linalg.norm(np_points - center, axis=1)
#     indices = distances.argsort()
#
#     take = min(len(indices), max_indices)
#     ret = indices[0:take]
#     return ret

def find_board(scene, scene_points_wb, debug=None):

    ret = {}
    ret['found'] = False

    if len(scene_points_wb) < 8:
        ret['message'] = 'too few wb points'
        return ret

    board_layout = create_board_layout(wb=True, bw=False, internal=True)
    board_layout = np.float32(board_layout).reshape(-1, 1, 2)

    board_wb_layouts = []
    for i in range(2, 7):
        for j in range(2, 7):
            if (i + j) % 2 == 0:
                test_layout = [(j - 1, i - 1),
                               (j + 1, i - 1),
                               (j + 1, i + 1),
                               (j - 1, i + 1)]
                board_wb_layouts.append(test_layout)

    scene_points = []
    scene_points.extend(scene_points_wb)
    # scene_points.extend(scene_points_bw)
    scene_points = np.array(scene_points)

    np_scene_points_wb = np.array(scene_points_wb)

    nn1 = NearestNeighbors(n_neighbors=1)
    nn1.fit(np_scene_points_wb)

    nnk = NearestNeighbors(n_neighbors=8)
    nnk.fit(np_scene_points_wb)

    center = np_scene_points_wb.mean(axis=0)
    distances, test_indices = nnk.kneighbors([center])

    match = None
    for chosen_idx in test_indices[0]:

        #chosen_idx = near_to_center_idxs[i]
        chosen = scene_points_wb[chosen_idx]

        distances, indices = nnk.kneighbors([chosen])
        nearest = np_scene_points_wb[indices].reshape(-1, 2)
        nearest = nearest[distances.argsort()].reshape(-1, 2)
        #utils.imshow("x", draw_points(scene, nearest.reshape(-1, 2), pause=True))

        quatern = find_bw_quatern(nearest)
        if len(quatern) < 2:
            continue
        qt_idx = [quatern[0][0], quatern[0][1], quatern[1][0], quatern[1][1]]
        scene_wb_candidate = order_points(nearest[qt_idx])

        if dbg(debug, 'test_layouts'):
            utils.imshow('idea chosen', draw_points(scene, [chosen], green))
            colors = [red, blue, magenta, yellow]
            utils.imshow('idea nn',  draw_points(scene, scene_wb_candidate, colors))
            cv2.waitKey()

        match = match_layout(scene,
                     board_layout,
                     board_wb_layouts,
                     nn1,
                     scene_wb_candidate,
                     match,
                     debug=debug
        )

    if match is None:
        ret['message'] = "board not found"
        return ret

    score = match['score']
    if score < 10:
        ret['message'] = 'too low confidence (score {})'.format(score)
        return ret

    # Now using the best H found so far, we will try to get a better one
    # using four other correspondences on the board
    # (but this time we will look for them to
    # be as far apart as possible from each other)
    improve_match(scene, nn1, match, scene_points)


    ret['found'] = True
    ret['match'] = match

    return ret


def warp_board(image, results, square_size=96):

    board_results = results['board_results']
    if not board_results['found']:
        return None

    match = board_results['match']
    H = match['H']
    board_layout = [[0, 0], [0, 8], [8, 8], [8, 0]]
    board_layout = np.float32(board_layout).reshape(-1, 1, 2)
    scene_board_boundings = cv2.perspectiveTransform(board_layout, H)

    square_size = 96

    H_inv = cv2.getPerspectiveTransform(scene_board_boundings, board_layout * square_size)
    dsize = (8 * square_size, 8 * square_size)
    warped = cv2.warpPerspective(image, H_inv, dsize)
    #vis = image.copy()
    #cv2.drawContours(vis, [scene_board_boundings.astype(int)], -1, green)
    #utils.imshow('warped', warped)
    #utils.imshow('scene', vis)
    return warped

def show_results(image, results):

    ret = {}
    utils.imshow('scene', image)

    vertices = results['vertices']
    utils.imshow('wb vertices', draw_points(image, vertices))

    board_results = results['board_results']
    if not board_results['found']:
        print(board_results['message'])
        return ret

    match = board_results['match']
    H = match['H']
    board_layout = [[0, 0], [0, 8], [8, 8], [8, 0]]
    board_layout = np.float32(board_layout).reshape(-1, 1, 2)
    scene_board_boundings = cv2.perspectiveTransform(board_layout, H)
    # map_to_scene(board_results, board_layout)
    #ret['board_boundings'] = scene_board_boundings.reshape(-1, 2)

    # board_layout = [[0, 0], [0, 8], [8, 8], [8, 0]]
    # board_layout = np.float32(board_layout).reshape(-1, 1, 2)
    # scene_board_boundings = cv2.perspectiveTransform(board_layout, H)

    square_size = 96

    H_inv = cv2.getPerspectiveTransform(scene_board_boundings, board_layout * square_size)
    dsize = (8 * square_size, 8 * square_size)
    warped = cv2.warpPerspective(image, H_inv, dsize)
    vis = image.copy()
    cv2.drawContours(vis, [scene_board_boundings.astype(int)], -1, green)
    utils.imshow('warped', warped)
    utils.imshow('found', vis)

    ret['warped'] = warped
    ret['found'] = vis

    return ret

class ChessBoardDetector:

    def get_default_debug(self):
        debug = {}
        debug['improve_subpix'] = False
        debug['match_layout'] = False
        debug['test_layouts'] = False
        return debug

    def __init__(self, debug=None):
        """
        """

        if debug is None:
            debug = self.get_default_debug()
        self._debug = debug

        # Creates synthetic patches
        patch_wb, patch_bw = create_patchs_synthetic(radius=16)
        self.patch_wb = patch_wb


    def detect(self, image):

        ret = {}
        ret['found'] = False

        points_wb = template_match(image, self.patch_wb, debug=self._debug)
        ret['vertices'] = points_wb

        board_results = find_board(image, points_wb, debug=self._debug)
        ret['board_results'] = board_results

        if not board_results['found']:
            ret['message'] = board_results['message']
        else:
            ret['found'] = True

        return ret




def detect_on_path(path='images/samples/', file_pattern="scene?.*g"):

    search = path + file_pattern

    files = [f for f in glob.glob(search, recursive=False)]
    files = sorted(files)

    detector = ChessBoardDetector()
    for file in files:

        print(file)

        scene = cv2.imread(file)
        #scene = utils.shrink_if_large(scene, max=640)
        scene = utils.shrink_if_large(scene)

        results = detector.detect(scene)
        ret = show_results(scene, results)
        for key in ret.keys():
            fn, ext = os.path.splitext(file)
            out_file = fn + "_" + key + ext
            cv2.imwrite(out_file, ret[key])

        k = cv2.waitKey()
        if k == ord('q'):
            break


    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_on_path()