from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Dict
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pyedflib


class SleepStage(Enum):
    AWAKE = 0
    NREM  = 1
    REM   = 2


@dataclass
class Config:
    fs: float
    epoch_sec: float
    eeg_band_theta: Tuple[float, float] = (6.0, 9.0)
    eeg_band_delta: Tuple[float, float] = (0.5, 4.0)
    emg_rms_floor: float = 1e-12
    z_thr_emg_awake: float = 0.5
    z_thr_td_rem: float = 0.5
    z_thr_delta_nrem: float = 0.5


@dataclass
class EpochFeatures:
    start_time: float
    end_time: float
    td_ratio: float
    delta_power: float
    theta_power: float
    emg_rms: float
    td_ratio_z: float = np.nan
    delta_z: float = np.nan
    emg_z: float = np.nan
    stage: Optional[SleepStage] = None


def load_edf_from_path(path: str, eeg_label: Optional[str] = None, emg_label: Optional[str] = None) -> pd.DataFrame:
    f = pyedflib.EdfReader(path)
    n_signals = f.signals_in_file
    labels = f.getSignalLabels()
    sfreqs = [f.getSampleFrequency(i) for i in range(n_signals)]

    eeg_lbl = eeg_label or next((lbl for lbl in labels if "EEG" in lbl.upper()), labels[0])
    emg_lbl = emg_label or next((lbl for lbl in labels if "EMG" in lbl.upper()), labels[-1])

    eeg_idx = labels.index(eeg_lbl)
    emg_idx = labels.index(emg_lbl)

    eeg = f.readSignal(eeg_idx)
    emg = f.readSignal(emg_idx)
    f.close()

    fs_eeg = sfreqs[eeg_idx]
    fs_emg = sfreqs[emg_idx]
    fs = float(np.mean([fs_eeg, fs_emg]))

    n_samples = min(len(eeg), len(emg))
    t = np.arange(n_samples) / fs
    return pd.DataFrame({"time": t, "eeg": eeg[:n_samples], "emg": emg[:n_samples]})


def to_epoch(x: np.ndarray, epoch_len: int) -> np.ndarray:
    n_epochs = len(x) // epoch_len
    return x[: n_epochs * epoch_len].reshape((n_epochs, epoch_len))


def zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd <= 0 or np.isnan(sd):
        return np.zeros_like(arr)
    return (arr - mu) / sd


def classify_epoch(feature: EpochFeatures, cfg: Config) -> SleepStage:
    """
    Very simple rule-based classifier on z-scored features:
      - AWAKE if EMG high and theta/delta not high
      - REM if theta/delta high and EMG low
      - NREM if delta high and EMG not high
    Tune cfg thresholds to your data.
    """
    if (feature.emg_z > cfg.z_thr_emg_awake) and (feature.td_ratio_z < cfg.z_thr_td_rem):
        return SleepStage.AWAKE
    if (feature.td_ratio_z > cfg.z_thr_td_rem) and (feature.emg_z < 0.0):
        return SleepStage.REM
    if (feature.delta_z > cfg.z_thr_delta_nrem) and (feature.emg_z < cfg.z_thr_emg_awake):
        return SleepStage.NREM
    # fallback: whichever looks most likely
    # choose by max of (delta_z -> NREM, td_ratio_z -> REM, emg_z -> AWAKE)
    scores = {
        SleepStage.NREM: feature.delta_z,
        SleepStage.REM: feature.td_ratio_z,
        SleepStage.AWAKE: feature.emg_z,
    }
    return max(scores, key=scores.get)


def extract_features_per_epoch(df: pd.DataFrame, cfg: Config) -> tuple[list[EpochFeatures], Dict[SleepStage, int]]:
    samples_per_epoch = int(round(cfg.epoch_sec * cfg.fs))
    eeg = to_epoch(df["eeg"].to_numpy(float), samples_per_epoch)
    emg = to_epoch(df["emg"].to_numpy(float), samples_per_epoch)
    n_epochs = min(eeg.shape[0], emg.shape[0])

    feats: List[EpochFeatures] = []
    for i in range(n_epochs):
        start_time = i * cfg.epoch_sec
        end_time = (i + 1) * cfg.epoch_sec

        emg_rms = float(np.sqrt(np.mean(emg[i] ** 2)))
        emg_rms = max(emg_rms, cfg.emg_rms_floor)

        freqs = np.fft.rfftfreq(eeg.shape[1], d=1 / cfg.fs)
        psd = np.abs(np.fft.rfft(eeg[i])) ** 2

        delta_mask = (freqs >= cfg.eeg_band_delta[0]) & (freqs <= cfg.eeg_band_delta[1])
        theta_mask = (freqs >= cfg.eeg_band_theta[0]) & (freqs <= cfg.eeg_band_theta[1])

        delta_power = float(np.sum(psd[delta_mask]))
        theta_power = float(np.sum(psd[theta_mask]))
        td_ratio = theta_power / (delta_power + 1e-12)

        feats.append(EpochFeatures(start_time, end_time, td_ratio, delta_power, theta_power, emg_rms))

    td_z = zscore(np.log10(np.array([f.td_ratio for f in feats])))
    delta_z = zscore(np.log10(np.array([f.delta_power for f in feats])))
    emg_z = zscore(np.log10(np.array([max(f.emg_rms, cfg.emg_rms_floor) for f in feats])))

    max_streaks = {SleepStage.REM: 0, SleepStage.NREM: 0, SleepStage.AWAKE: 0}
    current_streak = 0
    previous_stage: Optional[SleepStage] = None

    for i, f in enumerate(feats):
        f.td_ratio_z = float(td_z[i])
        f.delta_z = float(delta_z[i])
        f.emg_z = float(emg_z[i])
        f.stage = classify_epoch(f, cfg)

        if f.stage == previous_stage:
            current_streak += 1
        else:
            if previous_stage is not None:
                max_streaks[previous_stage] = max(max_streaks[previous_stage], current_streak)
            previous_stage = f.stage
            current_streak = 1

        if current_streak > max_streaks[f.stage]:
            max_streaks[f.stage] = current_streak

    if previous_stage is not None:
        max_streaks[previous_stage] = max(max_streaks[previous_stage], current_streak)

    return feats, max_streaks


def _fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _plot_hypnogram_dataurl(feats: List[EpochFeatures]) -> str:
    t = [f.start_time for f in feats]
    y = [f.stage.value for f in feats]
    fig = plt.figure(figsize=(10, 3))
    plt.step(t, y, where="post")
    plt.yticks([0, 1, 2], ["AWAKE", "NREM", "REM"])
    plt.xlabel("Time (s)")
    plt.ylabel("Stage")
    plt.title("Hypnogram")
    plt.tight_layout()
    return _fig_to_data_url(fig)


def _plot_feature_dataurl(feats: List[EpochFeatures], attr: str, ylabel: str, title: str) -> str:
    t = [f.start_time for f in feats]
    v = [getattr(f, attr) for f in feats]
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t, v)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return _fig_to_data_url(fig)


def run_analysis(
    edf_path: str,
    fs_override: Optional[float],
    epoch_sec: float,
    z_thr_emg_awake: float = 0.5,
    z_thr_td_rem: float = 0.5,
    z_thr_delta_nrem: float = 0.5,
    eeg_label: Optional[str] = None,
    emg_label: Optional[str] = None,
) -> dict:
    df  = load_edf_from_path(edf_path, eeg_label, emg_label)
    # Force fs from caller just like the CLI (args.fs). Fail fast if not given.
    if fs_override is None or fs_override <= 0:
        fs = float(1)
    else:
        fs = fs_override
    cfg = Config(
        fs=fs,
        epoch_sec=epoch_sec,
        z_thr_emg_awake=z_thr_emg_awake,
        z_thr_td_rem=z_thr_td_rem,
        z_thr_delta_nrem=z_thr_delta_nrem,
    )

    feats, streaks = extract_features_per_epoch(df, cfg)

    img_hypno = _plot_hypnogram_dataurl(feats)
    img_td     = _plot_feature_dataurl(feats, "td_ratio_z", "Z-scored", "Theta/Delta (z)")
    img_delta  = _plot_feature_dataurl(feats, "delta_z", "Z-scored", "Delta Power (z)")
    img_emg    = _plot_feature_dataurl(feats, "emg_z", "Z-scored", "EMG RMS (z)")

    # CSV as data URL
    csv_buf = BytesIO()
    pd.DataFrame([f.__dict__ for f in feats]).to_csv(csv_buf, index=False)
    csv_b64 = base64.b64encode(csv_buf.getvalue()).decode("ascii")
    csv_data_url = f"data:text/csv;base64,{csv_b64}"
 
    return {
        "fs_used": fs,
        "streaks": {k.name: int(v) for k, v in streaks.items()},
        "images": {
            "hypnogram": img_hypno,
            "td_z": img_td,
            "delta_z": img_delta,
            "emg_z": img_emg,
        },
        "csv": csv_data_url,
    }
