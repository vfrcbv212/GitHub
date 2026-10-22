from __future__ import annotations

import copy
import itertools
import math

import cv2
import numpy as np

# x along length (net x=9), y along width. Intersections, not arbitrary line clicks.
LANDMARKS = {f"x={x}, y={y}": [x, y] for x in (0, 6, 9, 12, 18) for y in (0, 9)}


def project_point(matrix, xy, on_floor=False):
    if not on_floor:
        raise ValueError("Гомография пола запрещена для мяча в воздухе и рук в прыжке")
    return _transform_points(matrix, [xy])[0].tolist()


def _points(value, label="точки"):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(f"{label}: нужны конечные координаты [x, y]")
    return points


def _matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Некорректная матрица гомографии")
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("Неустойчивая матрица гомографии")
    return matrix


def _transform_points(matrix, points):
    h, p = _matrix(matrix), _points(points)
    homogeneous = np.column_stack((p, np.ones(len(p)))) @ h.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("Точка слишком близка к горизонту преобразования")
    mapped = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(mapped).all():
        raise ValueError("Неустойчивое преобразование точки")
    return mapped


def nondegenerate(points):
    try:
        points = _points(points)
    except (ValueError, TypeError):
        return False
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return False
    if len(np.unique(points, axis=0)) != len(points):
        return False
    # Four correspondences cannot contain three collinear points.
    if len(points) == 4:
        for a, b, c in itertools.combinations(points, 3):
            if abs(np.cross(b - a, c - a)) < 1e-5:
                return False
    return True


def fit(image, world, control_image=None, control_world=None, lens_model=None):
    """Fit known floor correspondences in upright original pixel coordinates.

    Explicit lens parameters are applied before estimating the homography. Raw
    clicked pixels remain in image_points; no model of the phone lens is guessed.
    Independent control points are necessary for verified=True.
    """
    raw, b = _points(image), _points(world)
    lens_model = validate_lens_model(lens_model)
    a = lens_correct(raw, lens_model)
    if len(a) != len(b) or not nondegenerate(a) or not nondegenerate(b):
        raise ValueError("Недостаточные или вырожденные точки. Продолжайте в координатах видео/приблизительных зонах")
    if np.any(b < 0) or np.any(b > [18, 9]):
        raise ValueError("Опорные точки пола должны принадлежать корту 18×9 м")
    h, mask = cv2.findHomography(a, b, cv2.RANSAC if len(a) > 4 else 0, 0.15)
    if h is None or np.linalg.matrix_rank(h) < 3 or np.linalg.cond(h) > 1e10:
        raise ValueError("Неустойчивое преобразование: выберите распределённые по площадке точки")
    mapped = _transform_points(h, a)
    rmse = float(np.sqrt(np.mean(np.sum((mapped - b) ** 2, axis=1))))
    inliers = int(mask.sum())
    if inliers < 4 or rmse > 0.5:
        raise ValueError(f"Плохое совмещение: RMS {rmse:.3f} м, опорных точек {inliers}")
    rng = np.random.default_rng(42)
    deviations = []
    for _ in range(40):
        jitter_pixels = lens_correct(raw + rng.normal(0, 2, raw.shape), lens_model)
        jitter, _ = cv2.findHomography(jitter_pixels, b, 0)
        if jitter is not None:
            try:
                deviations.append(np.linalg.norm(_transform_points(jitter, a) - mapped, axis=1).max())
            except ValueError:
                pass
    if not deviations:
        raise ValueError("Калибровка неустойчива к смещению опорных точек")
    sensitivity = float(np.quantile(deviations, 0.95))
    controls = None
    ci, cw = [], []
    if control_image is not None and len(control_image):
        ci, cw = _points(control_image, "Контрольные точки изображения"), _points(control_world, "Контрольные точки пола")
        if len(ci) != len(cw) or np.any(cw < 0) or np.any(cw > [18, 9]):
            raise ValueError("Некорректные контрольные соответствия пола")
        if any(np.any(np.linalg.norm(b - p, axis=1) < 1e-7) for p in cw):
            raise ValueError("Контрольные точки должны быть независимыми и не повторять опорные")
        cp = _transform_points(h, lens_correct(ci, lens_model))
        controls = float(np.sqrt(np.mean(np.sum((cp - cw) ** 2, axis=1))))
    elif control_world is not None and len(control_world):
        raise ValueError("Для контрольных координат пола нужны соответствующие точки изображения")
    checked = controls is not None and controls <= 0.3 and sensitivity <= 0.4
    return {"matrix": h.tolist(), "image_points": raw.tolist(), "world_points": b.tolist(),
            "control_image_points": np.asarray(ci).tolist(), "control_world_points": np.asarray(cw).tolist(),
            "fit_rmse_m": rmse, "inliers": inliers, "click_sensitivity_2px_p95_m": sensitivity,
            "control_rmse_m": controls, "geometry": "court" if checked else "approximate",
            "verified": checked, "warning": "4 точки не доказывают точность; нужны независимые контрольные точки",
            "coordinate_space": "rotated_original", "homography_space": "undistorted_original" if lens_model else "upright_original",
            "lens_model": lens_model}


def fit_landmarks(correspondences, controls=(), lens_model=None):
    """Fit explicitly selected known line intersections, never arbitrary line clicks.

    Each record contains image=[x,y] and landmark=<one key from LANDMARKS>.
    Control landmarks must not repeat any fitting landmark.
    """
    def unpack(rows):
        image, world = [], []
        for row in rows:
            name = row.get("landmark")
            if name not in LANDMARKS:
                raise ValueError("Выберите известное пересечение линий из списка ориентиров")
            image.append(row["image"])
            world.append(LANDMARKS[name])
        return image, world
    image, world = unpack(correspondences)
    ci, cw = unpack(controls)
    result = fit(image, world, ci, cw, lens_model=lens_model)
    result["landmarks"] = [row["landmark"] for row in correspondences]
    result["control_landmarks"] = [row["landmark"] for row in controls]
    return result


def normalize(xy, side):
    if side not in ("left", "right"):
        return None
    x, y = xy
    return [x, y] if side == "left" else [18 - x, 9 - y]


def side_at(project, team_id, source_id, t):
    entries = [s for s in project.all("side") if s["team_id"] == team_id and s["source_id"] == source_id
               and s["start"] <= t <= s["end"]]
    return entries[-1]["side"] if entries else None


def within(polygon, point):
    return len(polygon or []) >= 3 and cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), False) >= 0


def validate_lens_model(model, image_size=None):
    """Accept only explicitly supplied calibrated parameters, in upright pixels."""
    if not model:
        return None
    try:
        kind, k, d = model["type"], np.asarray(model["K"], float), np.asarray(model["D"], float).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Для измеренной модели нужны type, K (3×3), D") from exc
    if kind not in ("pinhole", "fisheye"):
        raise ValueError("Укажите измеренную модель объектива: pinhole или fisheye")
    if (k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(d).all()
        or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10):
        raise ValueError("Некорректная K: нужны положительные фокусные расстояния без skew и конечные параметры")
    if len(d) not in ((4,) if kind == "fisheye" else (4, 5, 8, 12, 14)):
        raise ValueError("Неверное число коэффициентов D для выбранной модели объектива")
    if model.get("image_size") is not None:
        size = model["image_size"]
        if len(size) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in size):
            raise ValueError("Укажите размер калибровочного изображения [ширина, высота]")
        if image_size is not None and not np.allclose(size, image_size):
            raise ValueError("Размер изображения отличается от калибровки объектива; автоматический перенос K запрещён")
    result = copy.deepcopy(model)
    result.update(type=kind, K=k.tolist(), D=d.tolist())
    return result


def lens_correct(points, model):
    p = _points(points)
    model = validate_lens_model(model)
    if model is None:
        return p.copy()
    k, d = np.asarray(model["K"], float), np.asarray(model["D"], float)
    p = p.reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    if model["type"] == "pinhole":
        return cv2.undistortPointsIter(p, k, d, None, k, criteria).reshape(-1, 2)
    return cv2.fisheye.undistortPoints(p, k, d, P=k, criteria=criteria).reshape(-1, 2)


def lens_distort(points, model):
    """Undo lens_correct for drawing the court overlay on upright raw pixels."""
    p = _points(points)
    model = validate_lens_model(model)
    if model is None:
        return p.copy()
    k, d = np.asarray(model["K"], float), np.asarray(model["D"], float)
    normalized = np.column_stack((p, np.ones(len(p)))) @ np.linalg.inv(k).T
    normalized /= normalized[:, 2, None]
    rvec, tvec = np.zeros(3), np.zeros(3)
    if model["type"] == "pinhole":
        result, _ = cv2.projectPoints(normalized[:, None, :], rvec, tvec, k, d)
    else:
        result, _ = cv2.fisheye.projectPoints(normalized[:, None, :], rvec, tvec, k, d.reshape(4, 1))
    if not np.isfinite(result).all():
        raise ValueError("Неустойчивое преобразование объектива")
    return result.reshape(-1, 2)


def validate_polygon(polygon, image_size=None, label="Многоугольник"):
    p = _points(polygon, label)
    if len(p) < 3 or len(np.unique(p, axis=0)) != len(p) or abs(cv2.contourArea(p.astype(np.float32))) < 1e-6:
        raise ValueError(f"{label}: нужны как минимум три разные невырожденные вершины")
    if image_size is not None and (np.any(p < 0) or np.any(p > image_size)):
        raise ValueError(f"{label}: вершина вне изображения")
    def cross(a, b, c):
        return float(np.cross(b-a, c-a))
    def on_segment(a, b, c):
        return abs(cross(a, b, c)) < 1e-8 and np.all(c >= np.minimum(a,b)-1e-8) and np.all(c <= np.maximum(a,b)+1e-8)
    def intersects(a, b, c, d):
        ab1, ab2, cd1, cd2 = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
        return ((ab1*ab2 < 0 and cd1*cd2 < 0) or on_segment(a,b,c) or on_segment(a,b,d)
                or on_segment(c,d,a) or on_segment(c,d,b))
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if j == i+1 or (i == 0 and j == len(p)-1):
                continue
            if intersects(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]):
                raise ValueError(f"{label}: самопересечение или касание несоседних сторон")
    return p.tolist()


def _interval(value, label):
    start, end = value.get("start", 0), value.get("end")
    if (not isinstance(start,(int,float)) or not math.isfinite(start) or start < 0 or
        (end is not None and (not isinstance(end,(int,float)) or not math.isfinite(end) or end < start))):
        raise ValueError(f"{label}: некорректный интервал времени")
    return float(start), float(end) if end is not None else None


def _polygon_inside(inner, outer):
    """Check full edges, including crossings of a concave visible-floor boundary."""
    boundary = np.asarray(outer,float)
    for a,b in zip(np.asarray(inner,float),np.roll(np.asarray(inner,float),-1,axis=0)):
        if not within(outer,a):
            return False
        v = b-a
        parameters = [0.0,1.0]
        for c,d in zip(boundary,np.roll(boundary,-1,axis=0)):
            w = d-c
            denominator = float(np.cross(v,w))
            if abs(denominator) > 1e-10:
                t,u = float(np.cross(c-a,w))/denominator,float(np.cross(c-a,v))/denominator
                if -1e-8 <= t <= 1+1e-8 and -1e-8 <= u <= 1+1e-8:
                    parameters.append(min(1,max(0,t)))
            elif abs(float(np.cross(c-a,v))) < 1e-10:
                for p in (c,d):
                    t = float(np.dot(p-a,v)/np.dot(v,v))
                    if 0 <= t <= 1:
                        parameters.append(t)
        parameters = sorted(set(parameters))
        if any(not within(outer,a+(b-a)*(lo+hi)/2) for lo,hi in zip(parameters,parameters[1:])):
            return False
    return True


def _scope(profile, source_id=None, source_sha256=None):
    if not profile.get("source_id"):
        raise ValueError("Профиль ракурса должен быть связан с исходным файлом")
    if source_id is not None and profile["source_id"] != source_id:
        raise ValueError("Новый исходник/ракурс: старая калибровка не переносится автоматически")
    if source_sha256 is not None and profile.get("source_sha256") != source_sha256:
        raise ValueError("Контрольная сумма исходника отличается от профиля ракурса")


def validate_profile(profile, source_id=None, source_sha256=None, image_size=None):
    """Return a scoped, validated profile without promoting unchecked geometry.

    A legacy unchecked matrix may remain available as a draft overlay, but only
    a successful independent check grants floor metres. Approximate user zones
    have explicit team and source scope and never create metric coordinates.
    """
    result = copy.deepcopy(profile)
    _scope(result, source_id, source_sha256)
    size = image_size if image_size is not None else result.get("image_size")
    if size is not None:
        if len(size) != 2 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x <= 0 for x in size):
            raise ValueError("Укажите размер исходного повёрнутого изображения [ширина, высота]")
        if result.get("image_size") is not None and not np.allclose(result["image_size"], size):
            raise ValueError("Размер изображения не совпадает с профилем ракурса")
        result["image_size"] = list(size)
    result["start"], result["end"] = _interval(result,"Профиль")
    result["lens_model"] = validate_lens_model(result.get("lens_model"), size)
    for names_key, points_key in (("landmarks","world_points"),("control_landmarks","control_world_points")):
        if names_key in result:
            names, points = result[names_key], result.get(points_key,[])
            if (len(names) != len(points) or any(name not in LANDMARKS for name in names)
                or any(not np.allclose(LANDMARKS[name],point) for name,point in zip(names,points))):
                raise ValueError("Подписанные ориентиры не совпадают с известными точками корта")
    for key in ("visible_image","people_roi","ball_roi","net_region","referee_region"):
        if result.get(key):
            result[key] = validate_polygon(result[key], size, key)
    result["obstacles"] = [validate_polygon(p,size,"Преграда") for p in result.get("obstacles",[])]
    if result.get("net"):
        net = _points(result["net"],"Отрезок сетки")
        if len(net) != 2 or np.linalg.norm(net[0]-net[1]) < 1e-6:
            raise ValueError("Сетку укажите двумя разными точками, отдельно от пола")
    zones = []
    for zone in result.get("zones",[]):
        z = copy.deepcopy(zone)
        if isinstance(z.get("zone"),bool) or z.get("zone") not in range(1,7):
            raise ValueError("Приблизительная зона должна иметь номер 1–6")
        if not isinstance(z.get("team_id"),str) or not z["team_id"]:
            raise ValueError("Приблизительная зона должна быть связана с командой")
        z.setdefault("source_id",result["source_id"])
        if z["source_id"] != result["source_id"]:
            raise ValueError("Зона относится к другому исходнику/ракурсу")
        z["start"], z["end"] = _interval(z,"Зона")
        z["polygon"] = validate_polygon(z.get("polygon"),size,"Приблизительная зона")
        if result.get("visible_image") and not _polygon_inside(z["polygon"],result["visible_image"]):
            raise ValueError("Приблизительная зона выходит за обозначенный видимый пол")
        z["coordinate_space"] = "rotated_original"
        z["status"] = "user_labelled_approximate"
        zones.append(z)
    result["zones"] = zones
    if result.get('line_constraints'):
        from .line_geometry import validate_segments
        result['line_constraints']=validate_segments(result['line_constraints'],size)
    checked = False
    if result.get("matrix") is not None:
        _matrix(result["matrix"])
        if result["lens_model"] and result.get("homography_space") != "undistorted_original":
            raise ValueError("Для модели объектива требуется заново оценить гомографию в исправленных координатах")
        ci, cw = result.get("control_image_points"), result.get("control_world_points")
        if result.get('verified') and result.get('fit_kind')=='labelled_lines' and ci and cw:
            from .line_geometry import fit_from_lines,line_error
            check=fit_from_lines(result.get('line_constraints',[]),size,ci,cw,result['lens_model'])
            control_rmse=float(np.sqrt(np.mean(np.sum((_transform_points(result['matrix'],lens_correct(ci,result['lens_model']))-np.asarray(cw))**2,axis=1))))
            fit_rmse=line_error(result['matrix'],result['line_constraints'],result['lens_model'])
            checked=check['verified'] and control_rmse<=.3 and fit_rmse<=.5
            result.update(control_rmse_m=control_rmse,fit_rmse_m=fit_rmse,click_sensitivity_2px_p95_m=check['click_sensitivity_2px_p95_m'])
        if result.get("verified") and ci and cw and result.get("image_points") and result.get("world_points"):
            # Recompute rather than trusting a manually edited 'verified' flag.
            check = fit(result["image_points"],result["world_points"],ci,cw,lens_model=result["lens_model"])
            h = _matrix(result["matrix"])
            mapped = _transform_points(h,lens_correct(ci,result["lens_model"]))
            control_rmse = float(np.sqrt(np.mean(np.sum((mapped-np.asarray(cw))**2,axis=1))))
            fit_mapped = _transform_points(h,lens_correct(result["image_points"],result["lens_model"]))
            fit_rmse = float(np.sqrt(np.mean(np.sum((fit_mapped-np.asarray(result["world_points"]))**2,axis=1))))
            checked = check["verified"] and control_rmse <= 0.3 and fit_rmse <= 0.5
            result.update(control_rmse_m=control_rmse,fit_rmse_m=fit_rmse,
                          click_sensitivity_2px_p95_m=check["click_sensitivity_2px_p95_m"])
    result["verified"] = bool(checked)
    result["geometry"] = "court" if checked else "approximate" if zones else "video"
    result["coordinate_space"] = "rotated_original"
    result.setdefault("homography_space","upright_original")
    if result.get("matrix") is not None and not checked:
        result["warning"] = "Гомография доступна только как черновое наложение; метрические показатели отключены"
    return result


def map_floor(profile, xy, source_id, time, team_id=None, on_floor=False, geometry_valid=True):
    """Map an observed floor anchor. Unknowns never become metres or zone zero."""
    if not on_floor:
        raise ValueError("Плоскость пола неприменима к мячу в воздухе и рукам в прыжке")
    _scope(profile,source_id)
    point = _points([xy])[0]
    if not isinstance(time,(int,float)) or not math.isfinite(time):
        raise ValueError("Время должно быть конечным числом")
    start,end = _interval(profile,"Профиль")
    answer = {"geometry":"video","floor_point":None,"zone":None,"approximate":False,
              "calibration_id":profile.get("id"),"coordinate_space":"rotated_original","reason":None}
    if profile.get("image_size") is not None and (np.any(point<0) or np.any(point>profile["image_size"])):
        return answer | {"reason":"Точка вне границ исходного изображения; наблюдение недоступно"}
    if not geometry_valid:
        return answer | {"reason":"Сдвиг камеры или ненадёжное совмещение: пространственные показатели отключены"}
    if time < start or (end is not None and time > end):
        return answer | {"reason":"Точка вне временного интервала профиля ракурса"}
    if profile.get("visible_image") and not within(profile["visible_image"],point):
        return answer | {"reason":"Точка вне обозначенной видимой области пола"}
    if any(within(p,point) for p in profile.get("obstacles",[])):
        return answer | {"reason":"Постоянная преграда; наблюдение пола недоступно"}
    if profile.get("verified") and profile.get("geometry") == "court" and profile.get("matrix") is not None:
        # validate_profile must run once when loading/saving a profile. Keep this hot path cheap.
        corrected = lens_correct([point],profile.get("lens_model"))
        mapped = _transform_points(profile["matrix"],corrected)[0]
        if np.any(mapped < -1e-6) or np.any(mapped > np.array([18,9])+1e-6):
            return answer | {"geometry":"court","reason":"Точка пола вне границ корта 18×9 м"}
        mapped = np.clip(mapped,[0,0],[18,9])
        answer.update(geometry="court",floor_point=mapped.tolist(),coordinate_space="court_m")
        answer["uncertainty"] = {"control_rmse_m":profile.get("control_rmse_m"),
                                 "click_sensitivity_2px_p95_m":profile.get("click_sensitivity_2px_p95_m")}
    matches = []
    for z in profile.get("zones",[]):
        zstart,zend = _interval(z,"Зона")
        if (z.get("source_id",source_id) == source_id and z.get("team_id") == team_id
            and zstart <= time and (zend is None or time <= zend) and within(z["polygon"],point)):
            matches.append(z["zone"])
    unique = set(matches)
    if len(unique) == 1:
        answer.update(zone=next(iter(unique)),approximate=True)
        if answer["geometry"] != "court":
            answer["geometry"] = "approximate"
        answer["zone_reason"] = "Ручная приблизительная зона в пикселях текущего ракурса; это не метрическая проекция"
    elif len(unique) > 1:
        answer["zone_reason"] = "Пересечение противоречивых зон: зона неизвестна"
    if answer["floor_point"] is None and answer["zone"] is None:
        answer["reason"] = answer.get("zone_reason") or "Недостаточно проверенной геометрии или наблюдаемых размеченных зон"
    return answer


def court_to_image(profile, world_points, source_id=None):
    """Draw a floor-model overlay, including measured lens distortion.

    This also supports an unchecked matrix as a *draft overlay*. Callers must
    label that state and never use the drawn lines as proof of calibration.
    """
    _scope(profile,source_id)
    if profile.get("matrix") is None:
        raise ValueError("Нет гомографии даже для чернового наложения")
    pixels = _transform_points(np.linalg.inv(_matrix(profile["matrix"])),world_points)
    return lens_distort(pixels,profile.get("lens_model")).tolist()


class CameraGuard:
    """Static-feature geometry check. No calibration transfer on uncertain movement."""
    def __init__(self, reference):
        self.reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        self.reference = cv2.resize(self.reference, (640, 360))
        self.orb = cv2.ORB_create(1500)
        self.kp, self.desc = self.orb.detectAndCompute(self.reference, None)

    def check(self, rgb):
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (640, 360))
        kp, desc = self.orb.detectAndCompute(gray, None)
        if self.desc is None or desc is None:
            return {"state": "unknown", "reason": "нет устойчивых ориентиров"}
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]
        if len(good) < 25:
            return {"state": "unknown", "reason": "мало совпадений фоновых ориентиров"}
        a = np.float32([self.kp[m.queryIdx].pt for m in good])
        b = np.float32([kp[m.trainIdx].pt for m in good])
        h, mask = cv2.findHomography(a, b, cv2.RANSAC, 2)
        if h is None or mask.sum() < 20:
            return {"state": "unknown", "reason": "совмещение неустойчиво"}
        corners = np.float32([[[0, 0], [640, 0], [640, 360], [0, 360]]])
        shift = float(np.max(np.linalg.norm(cv2.perspectiveTransform(corners, h) - corners, axis=2)))
        return {"state": "shifted" if shift > 5 else "stable", "shift_at_640px": shift,
                "reason": "порог сдвига 5 px в кадре шириной 640; правило не откалибровано на ваших видео"}
