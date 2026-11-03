import sys

import gymnasium
import numpy as np
import torch

from src import config


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_numpy_and_torch_work_together():
    x = torch.from_numpy(np.arange(4.0))
    assert torch.equal(x * 2, torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64))


def test_gymnasium_api():
    env = gymnasium.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    env.close()
    assert obs.shape == (4,)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_seed_sets_do_not_overlap():
    val = set(config.VAL_SEEDS)
    test = set(config.TEST_SEEDS)
    assert not val & test
    assert config.TRAIN_SEED_START > max(val | test)


def test_root_is_repo_folder():
    assert (config.ROOT / "src" / "config.py").is_file()
