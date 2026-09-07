"""Shadow-load trained policy checkpoints without promoting them to control."""
from __future__ import annotations

from pathlib import Path

from policy.registry import ModelRegistry, ModelVersion


def _registry(tmp_path: Path, monkeypatch) -> ModelRegistry:
    import policy.registry as reg

    monkeypatch.setattr(reg, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(reg, "STATE_FILE", tmp_path / "policy_state.json")
    return ModelRegistry()


def test_shadow_checkpoint_picks_lowest_loss_not_latest(tmp_path, monkeypatch) -> None:
    r = _registry(tmp_path, monkeypatch)
    older = tmp_path / "policy_v0001.pt"
    newer = tmp_path / "policy_v0002.pt"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    r._state.versions = [
        ModelVersion(version=1, arch="mlp2_enc2", created_at=1.0,
                     validation_loss=0.20, shadow_win_rate=0.0, path=str(older)),
        ModelVersion(version=2, arch="gru_enc2", created_at=2.0,
                     validation_loss=0.31, shadow_win_rate=0.0, path=str(newer)),
    ]
    ckpt = r.get_shadow_checkpoint()
    assert ckpt is not None
    assert ckpt.version == 1
    assert ckpt.arch == "mlp2_enc2"
    assert r.get_active() is None


def test_shadow_checkpoint_skips_missing_files(tmp_path, monkeypatch) -> None:
    r = _registry(tmp_path, monkeypatch)
    present = tmp_path / "policy_v0003.pt"
    present.write_bytes(b"ok")
    r._state.versions = [
        ModelVersion(version=2, arch="mlp2", created_at=1.0,
                     validation_loss=0.10, shadow_win_rate=0.0,
                     path=str(tmp_path / "gone.pt")),
        ModelVersion(version=3, arch="gru", created_at=2.0,
                     validation_loss=0.40, shadow_win_rate=0.0, path=str(present)),
    ]
    ckpt = r.get_shadow_checkpoint()
    assert ckpt is not None
    assert ckpt.version == 3


def test_training_advisory_names_shadow_checkpoint_when_nothing_promoted(
    tmp_path, monkeypatch,
) -> None:
    r = _registry(tmp_path, monkeypatch)
    p = tmp_path / "policy_v0004.pt"
    p.write_bytes(b"ok")
    r._state.versions = [
        ModelVersion(version=4, arch="gru_enc2", created_at=1.0,
                     validation_loss=0.3085, shadow_win_rate=0.0, path=str(p)),
    ]
    adv = r.training_advisory()
    assert adv["has_better_candidate"] is True
    assert adv["active_version"] == 0
    assert adv["best_candidate_version"] == 4
    assert "NOT promoted" in adv["note"]


def test_boot_restore_source_pins_shadow_checkpoint_without_promote() -> None:
    """Lived: bounce ran default_untrained while v93 sat on disk."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("main.py").read_text()
    helper = src[src.find("def _pick_policy_checkpoint"):src.find("async def main")]
    assert "get_shadow_checkpoint" in helper
    assert "shadow_checkpoint" in helper
    restore = src[src.find("def _restore_active_policy_controller"):src.find("async def main")]
    assert "enable_feature" not in restore
    assert ".promote(" not in restore
    boot = src[src.find("restored_nn = _restore_active_policy_controller"):src.find("Policy pipeline: active")]
    assert 'f"v{_ver:04d}_shadow"' in boot
    assert "default_untrained" in boot
