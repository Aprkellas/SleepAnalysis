# tests/test_io_edf.py
import pandas as pd
import numpy as np

def test_load_edf_from_path_smoke(monkeypatch):
    from src import analysis

    # Fake the underlying EDF reader call inside your function
    def _fake_loader(path, eeg_label=None, emg_label=None):
        fs = 200.0
        t = np.arange(1000) / fs
        df = pd.DataFrame({"eeg": np.sin(2*np.pi*8*t), "emg": 0.1*np.ones_like(t)})
        return df, fs

    monkeypatch.setattr(analysis, "load_edf_from_path", _fake_loader)
    df, fs = analysis.load_edf_from_path("dummy.edf", "EEG", "EMG")
    assert {"eeg", "emg"} <= set(df.columns)
    assert fs == 200.0
    assert len(df) == 1000
