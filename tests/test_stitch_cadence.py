"""The worker's stitch must conform to the delivery cadence the same way the
backend does.

WAN 2.2 I2V generates at 16 fps. `-r 24` duplicates one frame in three instead
of resampling motion — 32% of the frames in the delivered Aigiri cut never
moved. This file is the worker-side half of that fix; the backend half is
`app/render/cadence.py` + `tests/test_cadence.py`. The two are hand-kept in sync
(this box cannot import the backend), so the parity is what is asserted here.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitch import _cadence_filter_chain, _normalize_cmd  # noqa: E402

# The Director's ruling of 2026-08-05, written out literally so the test cannot
# pass by rebuilding the string from the helper it is checking.
APPROVED_CHAIN_24 = (
    "minterpolate=fps=48:mi_mode=mci:me_mode=bidir:mc_mode=aobmc:vsbmc=1,"
    "tmix=frames=2:weights='1 1',"
    "fps=24"
)


def test_chain_matches_the_backend_verbatim():
    assert _cadence_filter_chain(24) == APPROVED_CHAIN_24


def test_normalize_resamples_instead_of_duplicating():
    cmd = _normalize_cmd("in.mp4", "out.mp4", 512, 288, 24)
    assert "-r" not in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.endswith(APPROVED_CHAIN_24)


def test_trim_still_applies_with_the_conform():
    cmd = _normalize_cmd("in.mp4", "out.mp4", 512, 288, 24, 1.0, 2.0)
    assert cmd[cmd.index("-ss") + 1] == "1.000"
    assert cmd[cmd.index("-t") + 1] == "2.000"
    assert _cadence_filter_chain(24) in cmd[cmd.index("-vf") + 1]
