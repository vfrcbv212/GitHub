import copy

import numpy as np
import pytest

from volleyvision.calibration import (court_to_image, fit, fit_landmarks, lens_correct,
    lens_distort, map_floor, validate_lens_model, validate_polygon, validate_profile)


def fitted_profile(lens=None):
    ideal = np.array([[100,100],[1900,100],[1900,1000],[100,1000]],float)
    world = [[0,0],[18,0],[18,9],[0,9]]
    raw = lens_distort(ideal,lens)
    controls = lens_distort([[700,550]],lens)
    return fit(raw,world,controls,[[6,4.5]],lens_model=lens) | {
        "id":"cal1","source_id":"source1","source_sha256":"hash1",
        "image_size":[2100,1200],"start":0,"end":20}


def approximate_profile():
    return {"id":"zones1","source_id":"source1","source_sha256":"hash1",
        "geometry":"approximate","verified":False,"image_size":[640,360],
        "visible_image":[[0,0],[640,0],[640,360],[0,360]],
        "zones":[{"zone":4,"team_id":"A","source_id":"source1","start":2,"end":10,
                  "polygon":[[10,10],[200,10],[200,200],[10,200]]}]}


def test_checked_court_profile_maps_floor_and_keeps_raw_controls():
    original=fitted_profile()
    profile=validate_profile(original,source_id="source1",source_sha256="hash1",image_size=[2100,1200])
    assert profile["verified"] and profile["geometry"]=="court"
    assert original["image_points"]==[[100,100],[1900,100],[1900,1000],[100,1000]]
    assert original["control_image_points"]==[[700,550]]
    result=map_floor(profile,[700,550],"source1",5,on_floor=True)
    assert result["floor_point"]==pytest.approx([6,4.5]) and result["zone"] is None
    assert not result["approximate"]
    assert court_to_image(profile,[[6,4.5]],"source1")[0]==pytest.approx([700,550])
    with pytest.raises(ValueError,match="воздухе"):
        map_floor(profile,[700,550],"source1",5,on_floor=False)


def test_unchecked_homography_does_not_create_metres():
    profile=fit([[0,0],[1800,0],[1800,900],[0,900]],[[0,0],[18,0],[18,9],[0,9]])
    assert not profile["verified"]
    profile.update(source_id="s",id="draft")
    normalized=validate_profile(profile)
    assert normalized["geometry"]=="video"
    assert map_floor(normalized,[300,200],"s",0,on_floor=True)["floor_point"] is None
    assert court_to_image(normalized,[[3,2]],"s")[0]==pytest.approx([300,200])
    # A manually set flag without independent data is not accepted on profile load.
    profile["verified"]=True
    assert validate_profile(profile)["verified"] is False


def test_known_landmarks_and_independent_controls():
    rows=[{"image":[0,0],"landmark":"x=0, y=0"},
          {"image":[1800,0],"landmark":"x=18, y=0"},
          {"image":[1800,900],"landmark":"x=18, y=9"},
          {"image":[0,900],"landmark":"x=0, y=9"}]
    result=fit_landmarks(rows,[{"image":[600,0],"landmark":"x=6, y=0"}])
    assert result["verified"] and result["control_landmarks"]==["x=6, y=0"]
    with pytest.raises(ValueError,match="известное пересечение"):
        fit_landmarks([{"image":[0,0],"landmark":"arbitrary line click"}])
    with pytest.raises(ValueError,match="независимыми"):
        fit_landmarks(rows,[rows[0]])
    result.update(source_id="s")
    result["landmarks"][0]="x=18, y=0"
    with pytest.raises(ValueError,match="Подписанные ориентиры"):
        validate_profile(result)


def test_approximate_zone_scope_interval_and_no_false_metres():
    profile=validate_profile(approximate_profile(),source_id="source1",source_sha256="hash1")
    result=map_floor(profile,[100,100],"source1",5,team_id="A",on_floor=True)
    assert result["geometry"]=="approximate" and result["zone"]==4
    assert result["approximate"] and result["floor_point"] is None
    for point,t,team in (([300,100],5,"A"),([100,100],1,"A"),([100,100],11,"A"),
                         ([100,100],5,"B"),([100,100],5,None)):
        unknown=map_floor(profile,point,"source1",t,team_id=team,on_floor=True)
        assert unknown["zone"] is None and unknown["floor_point"] is None
    with pytest.raises(ValueError,match="переносится"):
        map_floor(profile,[100,100],"new-source",5,team_id="A",on_floor=True)


def test_overlapping_approximate_zones_do_not_guess():
    profile=approximate_profile();zone=copy.deepcopy(profile["zones"][0]);zone["zone"]=3
    profile["zones"].append(zone)
    normalized=validate_profile(profile)
    result=map_floor(normalized,[100,100],"source1",5,team_id="A",on_floor=True)
    assert result["zone"] is None and "противоречивых" in result["reason"]


def test_profile_source_hash_and_resolution_are_not_reused():
    profile=approximate_profile()
    with pytest.raises(ValueError,match="переносится"):
        validate_profile(profile,source_id="new-source")
    with pytest.raises(ValueError,match="сумма"):
        validate_profile(profile,source_sha256="changed-hash")
    with pytest.raises(ValueError,match="Размер"):
        validate_profile(profile,image_size=[1280,720])
    profile["zones"][0]["source_id"]="other"
    with pytest.raises(ValueError,match="другому исходнику"):
        validate_profile(profile)


@pytest.mark.parametrize("kind,d",[("pinhole",[.09,-.015,.001,-.001,0]),("fisheye",[.025,-.002,.0001,0])])
def test_measured_lens_round_trip_and_floor_overlay(kind,d):
    model={"type":kind,"K":[[1500,0,1050],[0,1500,600],[0,0,1]],"D":d,"image_size":[2100,1200]}
    points=np.array([[120,140],[900,400],[1850,1000]],float)
    raw=lens_distort(points,model)
    assert np.allclose(lens_correct(raw,model),points,atol=1e-6)
    profile=validate_profile(fitted_profile(model),source_id="source1",image_size=[2100,1200])
    assert profile["verified"] and profile["homography_space"]=="undistorted_original"
    raw_point=lens_distort([[700,550]],model)[0]
    result=map_floor(profile,raw_point,"source1",5,on_floor=True)
    assert result["floor_point"]==pytest.approx([6,4.5],abs=1e-5)
    assert court_to_image(profile,[[6,4.5]],"source1")[0]==pytest.approx(raw_point,abs=1e-5)
    with pytest.raises(ValueError,match="Размер"):
        validate_lens_model(model,image_size=[4200,2400])


def test_invalid_lens_cannot_silently_reinterpret_legacy_matrix():
    model={"type":"pinhole","K":[[1000,0,320],[0,1000,180],[0,0,1]],"D":[0,0,0,0,0]}
    profile=fitted_profile();profile["lens_model"]=model;profile.pop("homography_space")
    with pytest.raises(ValueError,match="заново оценить"):
        validate_profile(profile)
    model["D"]=[0,0]
    with pytest.raises(ValueError,match="число коэффициентов"):
        validate_lens_model(model)


@pytest.mark.parametrize("polygon",[
    [[0,0],[10,10],[0,10],[10,0]], # crossing
    [[0,0],[5,0],[10,0]], # no area
    [[0,0],[10,0],[10,10],[0,0]], # duplicate
    [[0,0],[10,0],[float("nan"),10]],
])
def test_invalid_polygons_are_rejected(polygon):
    with pytest.raises(ValueError):validate_polygon(polygon)


def test_concave_visible_floor_does_not_make_hidden_notch_observable():
    profile=approximate_profile();profile["image_size"]=[10,10]
    profile["visible_image"]=[[0,0],[10,0],[10,10],[7,10],[7,3],[3,3],[3,10],[0,10]]
    profile["zones"][0]["polygon"]=[[1,1],[9,1],[9,9],[1,9]]
    with pytest.raises(ValueError,match="видимый пол"):
        validate_profile(profile)


def test_shift_obstacle_and_outside_visibility_disable_maps():
    profile=validate_profile(fitted_profile()|{"visible_image":[[100,100],[1900,100],[1900,1000],[100,1000]],
        "obstacles":[[[600,500],[800,500],[800,600],[600,600]]]})
    assert map_floor(profile,[700,550],"source1",5,on_floor=True)["floor_point"] is None
    assert map_floor(profile,[50,50],"source1",5,on_floor=True)["floor_point"] is None
    shifted=map_floor(profile,[1000,700],"source1",5,on_floor=True,geometry_valid=False)
    assert shifted["geometry"]=="video" and shifted["floor_point"] is None
    assert "Сдвиг" in shifted["reason"]
