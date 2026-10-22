"""Conservative, guided alignment of an already identified floor-line model.

This is an optional initial-setup proposal, not an automatic court detector.
White pixels can belong to players or reflections. The result remains a draft
without independent controls and is never evidence that an unseen floor is visible.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from .calibration import court_to_image, validate_profile,lens_correct,_transform_points


def _world_lines(samples=90):
    points, groups, axes = [], [], []
    lines = [([x, 0], [x, 9], "x") for x in (0, 6, 9, 12, 18)]
    lines += [([0, y], [18, y], "y") for y in (0, 9)]
    for group, (start, end, axis) in enumerate(lines):
        points.extend(np.linspace(start, end, samples).tolist())
        groups.extend([group] * samples)
        axes.append(axis)
    return np.asarray(points), np.asarray(groups), axes


def _transform(matrix, points):
    p = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if np.any(np.abs(p[:, 2]) < 1e-10):
        raise ValueError("Неустойчивое уточнение: линия близка к горизонту")
    return p[:, :2] / p[:, 2, None]


def refine_floor_lines(profile, rgb, source_id, max_shift_px=6.0):
    """Return ``(validated_profile, diagnostics)`` for a small alignment proposal.

    The existing known correspondences initialize the homography. A white-line
    distance field inside the explicitly visible floor (or anchor hull) proposes
    a small pixel warp. Original fit anchors regularize it. Independent control
    points are deliberately absent from the objective and are checked afterwards.
    Every decline returns the unchanged validated profile and a reason.

    ``applied`` means a draft alignment proposal passed these engineering guards;
    it does not mark any video event as confirmed or claim measured field accuracy.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Нужен RGB uint8 стоп-кадр исходного повёрнутого видео")
    if not isinstance(max_shift_px, (int, float)) or not np.isfinite(max_shift_px) or not 0 < max_shift_px <= 20:
        raise ValueError("Допустимое уточнение должно быть от 0 до 20 пикселей")
    height, width = image.shape[:2]
    original = validate_profile(profile, source_id=source_id, image_size=[width, height])
    diagnostics = {"applied": False, "method": "guided white-line distance + original-anchor regularisation",
        "before_median_distance_px": None, "after_median_distance_px": None,
        "before_mean_distance_px": None, "after_mean_distance_px": None,
        "visible_model_samples": 0, "supported_line_groups": 0,
        "maximum_shift_px": 0.0, "shift_limit_px": float(max_shift_px),
        "independent_controls_used_in_optimization": False,
        "white_pixel_count": 0, "roi_source": None,
        "limitations": ["Предложение по белым линиям, не детектор площадки",
            "Окклюзии и блики могут искажать результат; проверить наложение",
            "Невидимая область не становится наблюдаемой", "Качество на реальных видео не измерено"]}

    def decline(reason):
        return original, diagnostics | {"reason": reason}

    if original.get("matrix") is None:
        return decline("Сначала укажите известные ориентиры пола; матрица ещё не оценена")
    raw = np.asarray(original.get("image_points", []), dtype=float)
    worlds = np.asarray(original.get("world_points", []), dtype=float)
    if original.get('fit_kind')=='labelled_lines' and original.get('line_constraints'):
        raw=np.concatenate([np.asarray(s['image']) for s in original['line_constraints']])
        # These are regularization anchors of the initial model, not new measured
        # point correspondences or independent validation evidence.
        worlds=_transform_points(original['matrix'],lens_correct(raw,original.get('lens_model')))
    if raw.ndim != 2 or raw.shape[1:] != (2,) or len(raw) < 4 or worlds.shape != raw.shape:
        return decline("Для уточнения нужны сохранённые известные опорные соответствия")
    roi = np.zeros((height, width), np.uint8)
    if original.get("visible_image"):
        polygon = np.asarray(original["visible_image"], np.float32)
        diagnostics["roi_source"] = "user_visible_floor"
    else:
        polygon = cv2.convexHull(raw.astype(np.float32)).reshape(-1, 2)
        diagnostics["roi_source"] = "hull_of_known_floor_anchors"
    cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 255)
    for obstacle in original.get("obstacles", []):
        cv2.fillPoly(roi, [np.rint(obstacle).astype(np.int32)], 0)
    roi_area = int(np.count_nonzero(roi))
    if roi_area < 200:
        return decline("Слишком мала наблюдаемая область пола")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 150], np.uint8), np.array([179, 75, 255], np.uint8))
    white = cv2.bitwise_and(white, roi)
    white_count = int(np.count_nonzero(white))
    diagnostics["white_pixel_count"] = white_count
    if white_count < 30 or white_count / roi_area > 0.3:
        return decline("Недостаточно различимых белых линий или белая область слишком велика")
    distance = cv2.distanceTransform(255-white, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    world, groups, axes = _world_lines()
    initial_pixels = np.asarray(court_to_image(original, world, source_id), dtype=float)
    inside_image = ((initial_pixels[:, 0] >= 0) & (initial_pixels[:, 0] <= width-1) &
                    (initial_pixels[:, 1] >= 0) & (initial_pixels[:, 1] <= height-1))
    within_roi = np.zeros(len(world), bool)
    ii = np.flatnonzero(inside_image)
    if len(ii):
        rounded = np.rint(initial_pixels[ii]).astype(int)
        within_roi[ii] = roi[rounded[:, 1], rounded[:, 0]] != 0
    keep = inside_image & within_roi
    world, groups, initial_pixels = world[keep], groups[keep], initial_pixels[keep]
    diagnostics["visible_model_samples"] = len(world)
    if len(world) < 50:
        return decline("Слишком мало видимых отрезков модели внутри обозначенного пола")

    def distances(pixels):
        return map_coordinates(distance, [pixels[:, 1], pixels[:, 0]], order=1, output=np.float64,
                               mode="constant", cval=max(30.0, 4*max_shift_px))

    before = distances(initial_pixels)
    supported_axes = []
    for group in np.unique(groups):
        d = before[groups == group]
        if len(d) >= 10 and np.count_nonzero(d <= max_shift_px+2) >= max(10, .4*len(d)):
            supported_axes.append(axes[group])
    diagnostics.update(supported_line_groups=len(supported_axes),
        before_median_distance_px=float(np.median(before)), before_mean_distance_px=float(np.mean(before)))
    if len(supported_axes) < 3 or len(set(supported_axes)) < 2:
        return decline("Нужны различимые отрезки хотя бы трёх линий двух направлений")
    if np.quantile(before, .8) > max_shift_px+3:
        return decline("Начальное совмещение слишком далеко от видимых линий; поправьте ориентиры")
    if np.median(before) < .15 and np.mean(before) < .35:
        return decline("Модель уже совмещена с видимыми белыми линиями")

    h = np.asarray(original["matrix"], float)
    inverse = np.linalg.inv(h)
    initial_anchor_pixels = np.asarray(court_to_image(original, worlds, source_id), float)

    def candidate_profile(parameters):
        p = parameters
        warp = np.array([[1+p[0]/width, p[1]/height, p[2]],
                         [p[3]/width, 1+p[4]/height, p[5]],
                         [p[6]/width**2, p[7]/height**2, 1]], dtype=float)
        candidate = copy.deepcopy(original)
        candidate["matrix"] = np.linalg.inv(warp @ inverse).tolist()
        return candidate

    def residuals(parameters):
        candidate = candidate_profile(parameters)
        pixels = np.asarray(court_to_image(candidate, world, source_id), float)
        anchors = np.asarray(court_to_image(candidate, worlds, source_id), float)
        # Controls are intentionally not consulted here.
        return np.r_[distances(pixels), .75*(anchors-raw).ravel(), .1*parameters]

    bound = max_shift_px / 2
    optimization = least_squares(residuals, np.zeros(8), bounds=(-bound, bound),
        loss="soft_l1", f_scale=1.5, max_nfev=100, xtol=1e-7, ftol=1e-7, gtol=1e-7)
    candidate = candidate_profile(optimization.x)
    proposed_pixels = np.asarray(court_to_image(candidate, world, source_id), float)
    proposed_anchors = np.asarray(court_to_image(candidate, worlds, source_id), float)
    shifts = np.r_[np.linalg.norm(proposed_pixels-initial_pixels, axis=1),
                   np.linalg.norm(proposed_anchors-initial_anchor_pixels, axis=1)]
    maximum_shift = float(np.max(shifts))
    after = distances(proposed_pixels)
    diagnostics.update(after_median_distance_px=float(np.median(after)),
        after_mean_distance_px=float(np.mean(after)), maximum_shift_px=maximum_shift,
        optimizer_evaluations=int(optimization.nfev))
    if maximum_shift > max_shift_px + 1e-5:
        return decline("Предложенный сдвиг превысил допустимый предел; ориентиры сохранены")
    if np.mean(before)-np.mean(after) < .1 or np.mean(after) > .9*np.mean(before):
        return decline("Надёжного улучшения совмещения не найдено; исходная матрица сохранена")
    candidate = validate_profile(candidate, source_id=source_id, image_size=[width, height])
    if original["verified"] and not candidate["verified"]:
        return decline("Уточнение не прошло независимые контрольные точки; исходная матрица сохранена")
    diagnostics.update(applied=True, reason="Предложено небольшое уточнение по видимым линиям; проверьте наложение")
    candidate["line_refinement"] = copy.deepcopy(diagnostics)
    return candidate, diagnostics
