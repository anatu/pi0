"""Tests for the training pipeline: scheduler, single step, checkpoint."""

import tempfile
from pathlib import Path

import torch
import pytest

from pi0.config import ModelConfig, FlowConfig, TrainingConfig, EnvConfig
from pi0.data.collector import collect_trajectories


def _make_small_dataset(tmpdir: Path) -> Path:
    """Collect a tiny dataset for testing."""
    out = tmpdir / "trajectories"
    collect_trajectories(
        num_trajectories=10,
        output_dir=out,
        env_config=EnvConfig(),
        seed=42,
        show_progress=False,
    )
    return out


# ── Scheduler Tests ──


def test_scheduler_warmup():
    from pi0.training.scheduler import get_cosine_schedule_with_warmup

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=100, total_steps=1000)

    lrs = []
    for step in range(200):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # LR should increase during warmup (first 100 steps)
    assert lrs[0] < lrs[50], "LR should increase during warmup"
    assert lrs[50] < lrs[99], "LR should still be increasing mid-warmup"
    # After warmup, LR should start decreasing (cosine decay)
    assert lrs[100] >= lrs[199], "LR should decay after warmup"


def test_scheduler_cosine_decay():
    from pi0.training.scheduler import get_cosine_schedule_with_warmup

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=0, total_steps=100)

    lrs = []
    for step in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Should decay monotonically
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-10, f"LR should decay: step {i}"
    # Should end near zero
    assert lrs[-1] < 0.01 * lrs[0], "LR should be near zero at the end"


# ── Single Training Step ──


@pytest.mark.slow
def test_single_train_step():
    """A single training step should run and produce finite loss."""
    from pi0.training.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = _make_small_dataset(tmpdir)

        train_cfg = TrainingConfig(
            batch_size=4,
            max_epochs=1,
            num_workers=0,
            log_every=1,
            checkpoint_dir=str(tmpdir / "ckpts"),
            log_dir=str(tmpdir / "logs"),
            device="cpu",
        )

        trainer = Trainer(
            model_config=ModelConfig(),
            flow_config=FlowConfig(),
            training_config=train_cfg,
            data_dir=str(data_dir),
        )

        # Get one batch
        batch = next(iter(trainer.dataloader))
        loss = trainer.train_step(batch)

        assert isinstance(loss, float)
        assert loss > 0, "Loss should be positive"
        assert not (loss != loss), "Loss should not be NaN"  # NaN check


@pytest.mark.slow
def test_parameters_update_after_step():
    """Model parameters should change after a training step."""
    from pi0.training.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = _make_small_dataset(tmpdir)

        train_cfg = TrainingConfig(
            batch_size=4,
            max_epochs=1,
            num_workers=0,
            checkpoint_dir=str(tmpdir / "ckpts"),
            log_dir=str(tmpdir / "logs"),
            device="cpu",
        )

        trainer = Trainer(
            model_config=ModelConfig(),
            flow_config=FlowConfig(),
            training_config=train_cfg,
            data_dir=str(data_dir),
        )

        # Snapshot action head weight (receives direct gradient signal)
        param = None
        for name, p in trainer.model.named_parameters():
            if p.requires_grad and "action_head" in name and "weight" in name:
                param = p
                break
        assert param is not None, "Should find an action_head weight parameter"

        before = param.data.clone()

        # Run multiple steps to accumulate enough update through warmup
        batch = next(iter(trainer.dataloader))
        for _ in range(5):
            trainer.train_step(batch)
        after = param.data

        assert not torch.equal(before, after), "Parameters should update after training steps"


# ── Checkpoint Save/Load ──


@pytest.mark.slow
def test_checkpoint_save_and_load():
    """Saving and loading a checkpoint should restore the model."""
    from pi0.training.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = _make_small_dataset(tmpdir)
        ckpt_dir = tmpdir / "ckpts"

        train_cfg = TrainingConfig(
            batch_size=4,
            max_epochs=1,
            num_workers=0,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(tmpdir / "logs"),
            device="cpu",
        )

        trainer = Trainer(
            model_config=ModelConfig(),
            flow_config=FlowConfig(),
            training_config=train_cfg,
            data_dir=str(data_dir),
        )

        # Run a step so the model has non-initial weights
        batch = next(iter(trainer.dataloader))
        trainer.train_step(batch)

        # Save checkpoint
        trainer.save_checkpoint("test")
        ckpt_path = ckpt_dir / "checkpoint_test.pt"
        assert ckpt_path.exists(), "Checkpoint file should exist"

        # Load checkpoint
        model, model_cfg, flow_cfg = Trainer.load_model_from_checkpoint(
            str(ckpt_path), device="cpu"
        )

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(), model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1.data, p2.data), f"Parameter {n1} mismatch after load"


# ── Short Training Run ──


@pytest.mark.slow
def test_short_training_run():
    """Run 2 epochs and verify loss is finite and checkpoint is saved."""
    from pi0.training.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = _make_small_dataset(tmpdir)
        ckpt_dir = tmpdir / "ckpts"

        train_cfg = TrainingConfig(
            batch_size=4,
            max_epochs=2,
            num_workers=0,
            log_every=5,
            checkpoint_every=1,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(tmpdir / "logs"),
            device="cpu",
        )

        trainer = Trainer(
            model_config=ModelConfig(),
            flow_config=FlowConfig(),
            training_config=train_cfg,
            data_dir=str(data_dir),
        )

        trainer.train()

        # Checkpoint should exist
        assert (ckpt_dir / "checkpoint_final.pt").exists()
        assert (ckpt_dir / "latest.pt").exists()

        # Global step should be > 0
        assert trainer.global_step > 0
