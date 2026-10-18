from facemosaic.config import DEFAULT_CONFIG, DEFAULTS, load_config, resolve_for
from facemosaic.detectors import scale_plan


def test_profiles_match_by_path():
    cfg = load_config()
    cfg["profiles"] = {"wrist": {"match": ["*wrist*"], "detector": {"score_low": 0.1}}}
    eff, names = resolve_for(cfg, "ep_001/Wrist_cam.mp4")
    assert names == ["wrist"] and eff["detector"]["score_low"] == 0.1
    eff, names = resolve_for(cfg, "ep_001/head.mp4")
    assert names == [] and eff["detector"]["score_low"] == 0.2


def test_default_config_file_loads_and_profiles_match():
    cfg = load_config(DEFAULT_CONFIG)
    assert set(cfg["profiles"]) == {"wrist", "head", "third"}
    _, names = resolve_for(cfg, "videos/chunk-000/observation.images.wrist/episode_000000.mp4")
    assert names == ["wrist"]
    _, names = resolve_for(cfg, "episode_0001/third_person.mp4")
    assert names == ["third"]


def test_scale_plan_dedupes_and_never_upscales():
    scales = DEFAULTS["detector"]["scales"]
    assert scale_plan(2688, 1512, scales) == [(1280, 720, 0.0), (640, 360, 0.15), (320, 180, 0.15)]
    # 640x480 robot camera: 1280 is not upscaled, so it equals 640 and is de-duplicated (first entry, no padding, wins)
    assert scale_plan(640, 480, scales) == [(640, 480, 0.0), (320, 240, 0.15)]
