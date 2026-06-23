"""Self-sensing motion-gate: the predictor must tie persistence on still frames (no phantom motion).

Diagnosis (5-agent trace): a global ridge with a position term hallucinated range-proportional motion on
STILL frames (~5cm @ 0.5m, ~31cm @ 3m). On a ~79%-static room that "static tax" made the model WORSE than
the trivial 'no change' baseline and dragged the pooled headline below zero. The growth signal itself (the
DYNAMIC-frame skill) was always intact. The gate predicts a delta only where a sector actually moved.
"""

import numpy as np

from cognition.self_sensing import SelfSensingLoop, N_SECTORS, DYNAMIC_THRESH_M


def test_still_frame_ties_persistence():
    loop = SelfSensingLoop()
    # A W that hallucinates motion from the POSITION block (reproduces the bug: delta ∝ absolute range).
    loop._W = np.zeros((2 * N_SECTORS, N_SECTORS))
    loop._W[:N_SECTORS, :] = np.eye(N_SECTORS) * 0.1
    frame = [2.0] * N_SECTORS
    loop.observe(frame, 1.0)
    loop.observe(frame, 2.0)                      # identical frames => still (velocity 0)
    diff = loop._pending["pred"] - loop._pending["base"]
    assert np.allclose(diff, 0.0), ("phantom motion not gated", diff)   # pred == persistence


def test_moving_sector_still_predicted():
    loop = SelfSensingLoop()
    loop._W = np.zeros((2 * N_SECTORS, N_SECTORS))
    loop._W[N_SECTORS:, :] = np.eye(N_SECTORS) * 0.5   # velocity -> delta (the real signal)
    f1 = [2.0] * N_SECTORS
    f2 = [2.0] * N_SECTORS
    f2[3] = 2.10                                   # sector 3 moved +10cm (> 2cm DYNAMIC_THRESH_M)
    loop.observe(f1, 1.0)
    loop.observe(f2, 2.0)
    diff = np.asarray(loop._pending["pred"]) - np.asarray(loop._pending["base"])
    assert abs(diff[3]) > 0                         # the moving sector is predicted
    still = [i for i in range(N_SECTORS) if i != 3]
    assert np.allclose(diff[still], 0.0)            # every still sector ties persistence


def test_sub_threshold_jitter_is_gated():
    # Sub-2cm jitter (lidar noise) must be treated as still — no phantom prediction.
    loop = SelfSensingLoop()
    loop._W = np.ones((2 * N_SECTORS, N_SECTORS)) * 0.05
    f1 = [2.0] * N_SECTORS
    f2 = [2.0 + 0.005] * N_SECTORS                  # 5mm jitter, below the 2cm floor
    loop.observe(f1, 1.0)
    loop.observe(f2, 2.0)
    diff = np.asarray(loop._pending["pred"]) - np.asarray(loop._pending["base"])
    assert np.allclose(diff, 0.0)                   # jitter gated -> ties persistence
