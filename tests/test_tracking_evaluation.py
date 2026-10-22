import pytest

from volleyvision.evaluation import detections,events
from volleyvision.vision import Trackers,TemporalCandidates,DEFAULTS


def test_lost_ball_is_not_published_or_reused():
    tracker=Trackers(DEFAULTS,1000,1000,"segment1")
    d={"bbox":[100,100,110,110],"score":.9,"class_id":37}
    first=tracker.update([d],0,"ball")
    assert first and first[0]["observed"]
    for i in range(8):assert tracker.update([],(i+1)/15,"ball")==[]
    again=tracker.update([d],.6,"ball")
    assert again and again[0]["track_id"]!=first[0]["track_id"]


def test_temporal_candidates_require_sequence_and_never_point():
    t=TemporalCandidates();p=[{"bbox":[80,80,140,200],"track_id":"p"}]
    def b(x,y):return [{"bbox":[x-2,y-2,x+2,y+2]}]
    assert t.update(0,b(100,110),p,.15) is None
    assert t.update(.07,b(120,100),p,.15) is None
    result=t.update(.14,b(100,110),p,.15)
    assert result and "winner" not in result and "type" not in result
    t.update(.3,[],p,.15)
    assert t.update(.4,b(120,100),p,.15) is None


def test_event_evaluation_counts_misses_and_unknown_fields():
    gt=[{"source_id":"s","type":"attack","start":1,"player_id":"A"},
        {"source_id":"s","type":"attack","start":3,"player_id":"B"}]
    pr=[{"source_id":"s","type":"attack","start":1.1,"player_id":None},
        {"source_id":"s","type":"attack","start":5,"player_id":"B"}]
    result=events(gt,pr)
    assert result["events"]["attack"]["tp"]==1
    assert result["events"]["attack"]["fp"]==1
    assert result["events"]["attack"]["fn"]==1
    assert result["fields"]["player_id"]["accuracy_on_known"] is None
    assert result["fields"]["player_id"]["coverage_among_matched_events"]==0


def test_detection_evaluation_does_not_count_duplicate():
    a={"image_id":1,"class_id":37,"bbox":[0,0,10,10]}
    result=detections([a],[a,a])
    assert result["classes"]["37"]["tp"]==1 and result["classes"]["37"]["fp"]==1
