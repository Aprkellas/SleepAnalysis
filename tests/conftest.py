# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def fs():
    return 200.0  # Hz

@pytest.fixture
def epoch_sec():
    return 10.0

@pytest.fixture
def sample_df(fs):
    # 60 seconds of data: synthetic EEG/EMG
    duration = 60.0
    n = int(fs * duration)
    t = np.arange(n) / fs

    # EEG: mix of delta (1-4 Hz) and theta (6-9 Hz)
    eeg = (0.8*np.sin(2*np.pi*2.0*t) + 0.3*np.sin(2*np.pi*7.5*t)).astype(float)

    # EMG: low baseline with occasional bursts
    emg = (0.05*np.random.default_rng(0).normal(size=n)).astype(float)
    emg[fs*20:fs*22] += 0.6  # “awake-like” burst between 20-22s

    df = pd.DataFrame({"eeg": eeg, "emg": emg})
    return df
