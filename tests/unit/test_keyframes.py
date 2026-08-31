"""src/keyframes.py — the selection rule. No ffmpeg, no model, no network.

`still_runs`, `keyframe` and `spans` are pure and take numbers, which is why they are tested
here on lists of floats rather than on a machine with ffmpeg and a corpus. `scene_scores` runs
the subprocess and is the gate's business; only its argv builder is checked here.

Why the rule is worth this many tests: it is the whole cost lever of VRAG-023. A rule that
selects one stretch too few loses a slide silently, and a rule that selects two-frame
coincidences stops discriminating between a screen-share and a documentary — measured, at
threshold 0.02 dropping min_run_frames to 2 takes video 611 from 7 stretches to 41.
"""

from __future__ import annotations

import pytest

from src import keyframes as mod


# --------------------------------------------------------------------------- still_runs


def test_no_scores_is_no_runs():
    assert mod.still_runs([], 0.02, 3) == []


def test_all_moving_selects_nothing():
    """A documentary with a cut every sample has no slide on screen."""
    assert mod.still_runs([0.0, 0.5, 0.6, 0.4, 0.7], 0.02, 3) == []


def test_all_still_is_one_run_covering_every_frame():
    scores = [0.0, 0.001, 0.001, 0.001, 0.001]
    assert mod.still_runs(scores, 0.02, 3) == [(0, 4)]


def test_a_run_covers_the_frame_before_its_first_low_score():
    """scores[i] compares frame i to frame i-1, so k low scores cover k+1 frames.

    Getting this wrong shortens every span by one sample and loses the moment the slide
    actually appeared — the frame the viewer would want to be seeked to.
    """
    #        idx 0     1     2      3      4     5
    scores = [0.0, 0.90, 0.90, 0.001, 0.001, 0.90]
    # low scores at 3 and 4 compare frames 2-3 and 3-4, so the still stretch is frames 2..4
    assert mod.still_runs(scores, 0.02, 3) == [(2, 4)]


def test_frame_zero_is_not_counted_as_still():
    """ffmpeg prints 0.0 for frame 0, which has no predecessor. It is not a measurement.

    Counting it would make every video begin with a still stretch whether or not anything was
    holding still, and the first frame of a music video is not a slide.
    """
    scores = [0.0, 0.90, 0.90, 0.90]
    assert mod.still_runs(scores, 0.02, 3) == []


def test_a_run_that_reaches_the_last_frame_is_closed():
    """A stretch at the end is never terminated by a high score, so the loop has to close it.

    Forgetting this drops the final slide of every deck that ends on one — and a deck ending
    on its conclusions slide is the common case.
    """
    scores = [0.0, 0.90, 0.001, 0.001, 0.001]
    assert mod.still_runs(scores, 0.02, 3) == [(1, 4)]


def test_min_frames_rejects_a_short_stretch():
    scores = [0.0, 0.90, 0.001, 0.90, 0.90]  # one low score => 2 frames covered
    assert mod.still_runs(scores, 0.02, 3) == []
    assert mod.still_runs(scores, 0.02, 2) == [(1, 2)]


def test_min_frames_below_two_is_refused():
    """A stretch of one frame has no score to compare with, so it cannot be measured still."""
    with pytest.raises(ValueError, match="at least 2"):
        mod.still_runs([0.0, 0.001], 0.02, 1)


def test_threshold_is_exclusive():
    """`< threshold`, not `<=`. A score exactly at the lever is not below it."""
    assert mod.still_runs([0.0, 0.02, 0.02, 0.02], 0.02, 3) == []
    assert mod.still_runs([0.0, 0.019, 0.019, 0.019], 0.02, 3) == [(0, 3)]


def test_several_runs_are_separated_by_movement():
    scores = [0.0, 0.001, 0.001, 0.90, 0.90, 0.001, 0.001, 0.001, 0.90]
    assert mod.still_runs(scores, 0.02, 3) == [(0, 2), (4, 7)]


def test_a_looser_threshold_stops_discriminating():
    """The measured failure the config comment records, in miniature.

    A slow pan reads as still once the threshold is loose enough, and then a documentary
    selects as many stretches as a screen-share does.
    """
    pan = [0.0] + [0.03] * 6  # a gentle drift, not a slide
    assert mod.still_runs(pan, 0.02, 3) == []
    assert mod.still_runs(pan, 0.05, 3) == [(0, 6)]


# --------------------------------------------------------------------------- keyframe


def test_keyframe_is_the_last_frame_of_the_run():
    """A slide is often still being drawn when the stretch begins and complete by its end."""
    assert mod.keyframe((4, 9)) == 9


def test_keyframe_of_a_two_frame_run():
    assert mod.keyframe((4, 5)) == 5


# --------------------------------------------------------------------------- spans


def frames(*times: float) -> list[dict]:
    return [
        {"file": f"frame_{i + 1:05d}.jpg", "t_s": t, "bytes": 1000}
        for i, t in enumerate(times)
    ]


def test_spans_are_measured_off_recorded_timestamps():
    """Off ingest's `t_s`, never off the sampling grid.

    Same rule src.chunk follows: the grid says where a sample was asked for, `t_s` is the
    pts_time of the frame ffmpeg actually kept, and on a variable-frame-rate source they
    differ.
    """
    recorded = frames(0.0, 5.005, 10.01, 15.015, 20.02)
    assert mod.spans([(1, 3)], recorded, 5.0) == [(5.005, 20.015)]


def test_a_span_ends_one_step_after_the_last_frame():
    """The slide is on screen until the next sample proves otherwise, not only when photographed."""
    recorded = frames(0.0, 5.0, 10.0)
    (start, end), = mod.spans([(0, 2)], recorded, 5.0)
    assert (start, end) == (0.0, 15.0)


def test_spans_never_run_backwards():
    """A zero or negative step must not produce a span schemas.caption would reject."""
    recorded = frames(0.0, 5.0)
    (start, end), = mod.spans([(1, 1)], recorded, -10.0)
    assert end >= start


def test_no_runs_is_no_spans():
    assert mod.spans([], frames(0.0, 5.0), 5.0) == []


# --------------------------------------------------------------------------- scene_args


def test_scene_args_measures_rather_than_filters():
    """`gte(scene,0)` is always true: the filter is an instrument, nothing is dropped.

    And `-loglevel info`, because metadata=print writes at info level — quieting ffmpeg here
    would silence the only output the function wants.
    """
    from pathlib import Path

    argv = mod.scene_args(Path("runs/x/frames"), "jpg")
    assert argv[0] == "ffmpeg"
    assert "select='gte(scene,0)',metadata=print" in argv
    assert "info" in argv
    assert "frame_%05d.jpg" in " ".join(argv)


def test_scene_args_names_the_configured_format():
    from pathlib import Path

    argv = mod.scene_args(Path("runs/x/frames"), "png")
    assert "frame_%05d.png" in " ".join(argv)
