"""scripts/deploy_addon.py: what it stages, where it backs up, how it copies.

The Supervisor builds the add-on from whatever sits in
``\\\\<ha>\\addons\\battery_optimizer``, so the stage IS the release. Pinned:
the build context is complete and self-contained, the run script reaches the
container with LF endings, the add-on version is APP_VERSION (so the version
the Supervisor shows is the version the app logs), backups can never land
where the Supervisor looks for add-ons, and a copy prunes stale files and
proves every byte.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "deploy_addon", REPO / "scripts" / "deploy_addon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = _load()


@pytest.fixture
def staged(tmp_path):
    return deploy.stage(tmp_path / "stage")


def test_stage_is_a_complete_build_context(staged):
    files = deploy.file_hashes(staged)
    for required in deploy.REQUIRED_STAGE_FILES:
        assert required in files
    lib = sorted(p.name for p in (REPO / "appdaemon/apps/battery_optimizer_lib").glob("*.py"))
    shipped = sorted(Path(f).name for f in files if f.startswith("app/battery_optimizer_lib/"))
    assert shipped == lib
    assert not any("__pycache__" in f for f in files)
    # The Dockerfile only COPYs what is in the stage.
    dockerfile = (staged / "Dockerfile").read_text()
    for line in dockerfile.splitlines():
        if line.startswith("COPY "):
            source = line.split()[1].rstrip("/")
            assert (staged / source).exists(), line


def test_stage_stamps_app_version(staged):
    manifest = yaml.safe_load((staged / "config.yaml").read_text(encoding="utf-8"))
    assert manifest["version"] == deploy.app_version()
    assert manifest["slug"] == "battery_optimizer"


def test_container_scripts_have_lf_endings(staged):
    for name in deploy.LF_FILES:
        assert b"\r\n" not in (staged / name).read_bytes(), name


def test_base_image_is_pinned():
    dockerfile = (REPO / "addon/battery_optimizer/Dockerfile").read_text()
    froms = [l for l in dockerfile.splitlines() if l.startswith("FROM ")]
    assert len(froms) == 1
    image = froms[0].split()[1]
    assert "ARG BUILD_FROM" not in dockerfile and "$BUILD_FROM" not in dockerfile
    assert ":" in image and not image.endswith(":latest")


def test_backups_never_live_under_addons(tmp_path):
    root = tmp_path / "share_root"
    assert deploy.addons_dir(root).parent == root / "addons"
    assert "addons" not in deploy.backup_root(root).relative_to(root).parts


def test_sync_prunes_stale_files_and_verifies(staged, tmp_path):
    dest = tmp_path / "addons" / "battery_optimizer"
    (dest / "app" / "battery_optimizer_lib" / "__pycache__").mkdir(parents=True)
    (dest / "app" / "battery_optimizer_lib" / "removed_module.py").write_text("x = 1\n")
    (dest / "app" / "battery_optimizer_lib" / "__pycache__" / "x.pyc").write_bytes(b"0")
    pruned = deploy.sync(staged, dest)
    assert "app/battery_optimizer_lib/removed_module.py" in pruned
    assert not (dest / "app" / "battery_optimizer_lib" / "__pycache__").exists()
    assert deploy.file_hashes(dest) == deploy.file_hashes(staged)


def test_backup_keeps_the_newest(tmp_path):
    dest = tmp_path / "addons" / "battery_optimizer"
    dest.mkdir(parents=True)
    (dest / "config.yaml").write_text("version: x\n")
    root = tmp_path / "share" / "battery_optimizer_backups"
    root.mkdir(parents=True)
    for i in range(7):
        (root / f"addon-2026010{i}-000000").mkdir()
    saved = deploy.backup(dest, root, keep=5)
    kept = sorted(p.name for p in root.iterdir())
    assert len(kept) == 5 and saved.name in kept


def test_log_check():
    v = "2026-10-05.1"
    ok = f"... Battery Optimizer version {v}: orchestrator=/app/battery_optimizer.py"
    assert deploy.check_log(ok, v) == []
    assert deploy.check_log(ok + "\nTraceback (most recent call last):", v)
    assert deploy.check_log("starting", v) == [f"no 'Battery Optimizer version {v}' line yet"]


def test_seed_plan_maps_config_paths_between_containers(tmp_path):
    plan = deploy.seed_plan({
        "load_profile_file": "/config/load_profile.json",
        "learning_data_file": "/config/battery_learning_data.json",
    }, tmp_path)
    by_key = {k: (s, d) for k, s, d in plan}
    assert set(by_key) == {"load_profile_file", "learning_data_file",
                           "prediction_tracker_file", "pv_profile_file"}
    src, dst = by_key["learning_data_file"]
    assert src == tmp_path / "addon_configs" / deploy.APPDAEMON_SLUG / "battery_learning_data.json"
    assert dst == tmp_path / "addon_configs" / deploy.SLUG / "battery_learning_data.json"


def test_seed_plan_skips_unset_and_rejects_foreign_paths(tmp_path):
    plan = deploy.seed_plan({"learning_data_file": ""}, tmp_path)
    assert "learning_data_file" not in {k for k, _, _ in plan}
    with pytest.raises(deploy.DeployError):
        deploy.seed_plan({"pv_profile_file": "/homeassistant/pv.json"}, tmp_path)
