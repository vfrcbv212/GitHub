import copy

import cv2
import numpy as np
import pytest

from volleyvision.calibration import court_to_image, fit, validate_profile
from volleyvision.line_refinement import refine_floor_lines


def scene(shift=(2, 1), controls=False, lens=None):
    width, height = 720, 420
    actual = np.array([[60, 60], [660, 60], [660, 360], [60, 360]], float)
    worlds = [[0, 0], [18, 0], [18, 9], [0, 9]]
    from volleyvision.calibration import lens_distort
    raw = lens_distort(actual+shift, lens)
    control_image = lens_distort([[360, 210]], lens).tolist() if controls else []
    true = fit(lens_distort(actual, lens), worlds, lens_model=lens) | {"source_id":"s","image_size":[width,height]}
    profile = fit(raw, worlds, control_image, [[9, 4.5]] if controls else [], lens_model=lens)
    # A slightly extended floor region keeps both manual and true line edges visible.
    profile.update(source_id="s",image_size=[width,height],visible_image=[[45,45],[675,45],[675,375],[45,375]])
    rgb = np.full((height, width, 3), (215, 115, 50), np.uint8)
    for a,b in [([x,0],[x,9]) for x in (0,6,9,12,18)]+[([0,y],[18,y]) for y in (0,9)]:
        points=np.rint(court_to_image(true,np.linspace(a,b,200),"s")).astype(np.int32)
        cv2.polylines(rgb,[points],False,(250,250,250),2,cv2.LINE_AA)
    return profile,rgb


def test_guided_alignment_improves_known_synthetic_lines_without_creating_verification():
    profile,rgb=scene()
    before=copy.deepcopy(profile)
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert diag["applied"]
    assert diag["after_mean_distance_px"]<diag["before_mean_distance_px"]
    assert diag["maximum_shift_px"]<=diag["shift_limit_px"]
    assert result["verified"] is False and result["geometry"]=="video"
    assert result["image_points"]==before["image_points"]
    assert profile==before  # the original profile and clicks are never modified


def test_independent_controls_are_preserved_and_checked_not_optimized():
    profile,rgb=scene(controls=True)
    assert profile["verified"]
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert diag["applied"]
    assert diag["independent_controls_used_in_optimization"] is False
    assert result["control_image_points"]==profile["control_image_points"]
    assert result["control_world_points"]==profile["control_world_points"]
    assert result["verified"] and validate_profile(result)["verified"]
    assert result["control_rmse_m"]<=profile["control_rmse_m"]


def test_no_lines_and_outside_wall_lines_leave_matrix_unchanged():
    profile,rgb=scene()
    rgb[:]=(80,110,50)
    # Strong bright objects outside the user-declared floor cannot refine it.
    cv2.rectangle(rgb,(0,0),(719,30),(255,255,255),-1)
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert not diag["applied"] and diag["white_pixel_count"]==0
    assert result["matrix"]==profile["matrix"]


def test_single_line_cannot_support_free_refinement():
    profile,rgb=scene()
    rgb[:]=(80,110,50)
    cv2.line(rgb,(60,60),(660,60),(255,255,255),2)
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert not diag["applied"]
    assert "трёх" in diag["reason"]
    assert result["matrix"]==profile["matrix"]


def test_already_aligned_and_large_initial_error_do_not_force_adjustment():
    profile,rgb=scene(shift=(0,0))
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert not diag["applied"]
    assert result["matrix"]==profile["matrix"]
    profile,rgb=scene(shift=(25,20))
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert not diag["applied"]
    assert result["matrix"]==profile["matrix"]


def test_explicit_lens_model_is_used_and_not_guessed():
    model={"type":"pinhole","K":[[900,0,360],[0,900,210],[0,0,1]],"D":[.05,-.01,.001,0,0],"image_size":[720,420]}
    profile,rgb=scene(shift=(2,1),controls=True,lens=model)
    result,diag=refine_floor_lines(profile,rgb,"s")
    assert result["lens_model"]==profile["lens_model"]
    assert result["homography_space"]=="undistorted_original"
    assert diag["maximum_shift_px"]<=diag["shift_limit_px"]


def test_scope_and_image_type_are_enforced():
    profile,rgb=scene()
    with pytest.raises(ValueError,match="переносится"):
        refine_floor_lines(profile,rgb,"another-source")
    with pytest.raises(ValueError,match="RGB uint8"):
        refine_floor_lines(profile,rgb.astype(float),"s")
    with pytest.raises(ValueError):refine_floor_lines(profile,rgb,"s",max_shift_px=100)
