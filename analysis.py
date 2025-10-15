from typing import Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from io import BytesIO

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import base64

class SleepStage(Enum):
    AWAKE = 0
    NREM = 1    # Non-REM 
    REM = 2

@dataclass
class Config:
    fs: float
    epoch_sec: float
    eeg_band_theta: Tuple[float, float] = (6.0, 9.0)      # common rat theta band
    eeg_band_delta: Tuple[float, float] = (0.5, 4.0)      # delta band
    emg_rms_floor: float = 1e-12                          # avoid zeros
    # Z-thresholds for staging (tweak per dataset)
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

def run_analysis(
    edf_path: str,
    fs_override: Optional[float],
    epoch_sec: float,
    out_dir: Path,
    z_thr_emg_awake: float = 0.5,
    z_thr_td_rem: float = 0.5,
    z_thr_delta_nrem: float = 0.5,
    eeg_label: Optional[str] = None,
    emg_label: Optional[str] = None,
    ) -> dict:
    df, inferred_fs = load_edf_from_path(edf_path, eeg_label, emg_label)
    fs = fs_override if fs_override and fs_override > 0 else inferred_fs

    cfg = Config(
        fs=fs,
        epoch_sec=epoch_sec,
        z_thr_emg_awake=z_thr_emg_awake,
        z_thr_td_rem=z_thr_td_rem,
        z_thr_delta_nrem=z_thr_delta_nrem,
    )

    feats, streaks = extract_features_per_epoch(df, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_hypno = out_dir / "hypnogram.png"
    img_td     = out_dir / "theta_delta_z.png"
    img_delta  = out_dir / "delta_z.png"
    img_emg    = out_dir / "emg_z.png"

    img_hypno  = plot_hypnogram(feats, img_hypno)
    img_td     = plot_feature_dataurl(feats, "td_ratio_z", "Z-scored", "Theta/Delta (z)")
    img_delta  = plot_feature_dataurl(feats, "delta_z", "Z-scored", "Delta Power (z)")
    img_emg    = plot_feature_dataurl(feats, "emg_z", "Z-scored", "EMG RMS (z)")

    return {
        "fs_used": fs,
        "streaks": {k.name: int(v) for k, v in streaks.items()},
        "images": {
            "hypnogram": str(img_hypno.name),
            "td_z": str(img_td.name),
            "delta_z": str(img_delta.name),
            "emg_z": str(img_emg.name),
        },
    }

def fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def load_ascii(
    args
) -> pd.DataFrame:
    """
    Load ASCII into DataFrame with columns 'eeg' and 'emg'.
    Supports either named columns (header) or index-based selection (no header).
    If delimiter == 'whitespace' it uses delim_whitespace=True.
    """
    path = args.input
    if args.delimiter == "whitespace":
        df = pd.read_csv(path, delim_whitespace=True, header=0 if args.has_header else None)
    else:
        df = pd.read_csv(path, sep=args.delimiter, header=0 if args.has_header else None)

    if args.has_header:
        if args.eeg_col is None or args.emg_col is None:
            raise ValueError("When --has-header, you must pass --eeg-col and --emg-col names.")
        eeg = df[args.eeg_col].to_numpy(dtype=float)
        emg = df[args.emg_col].to_numpy(dtype=float)
    else:
        if args.eeg_col_idx is None or args.emg_col_idx is None:
            raise ValueError("When --no-header, you must pass --eeg-col-idx and --emg-col-idx (0-based).")
        eeg = df.iloc[:, args.eeg_col_idx].to_numpy(dtype=float)
        emg = df.iloc[:, args.emg_col_idx].to_numpy(dtype=float)

    out = pd.DataFrame({"eeg": eeg, "emg": emg})
    return out

def load_edf_from_path(args) -> pd.DataFrame:
    """
    Load EEG/EMG data from an EDF or EDF+ file into a pandas DataFrame
    with columns 'eeg' and 'emg'.

    Expected EDF channel labels: something like 'EEG', 'EEG_dominant_freq', 'EMG', etc.
    """

    path = args.input
    f = pyedflib.EdfReader(path)

    # --- Read metadata
    n_signals = f.signals_in_file
    labels = f.getSignalLabels()
    sfreqs = [f.getSampleFrequency(i) for i in range(n_signals)]

    # --- Choose channels
    # Allow explicit user selection or try to auto-match
    eeg_label = args.eeg_col or next((lbl for lbl in labels if "EEG" in lbl.upper()), labels[0])
    emg_label = args.emg_col or next((lbl for lbl in labels if "EMG" in lbl.upper()), labels[-1])

    eeg_idx = labels.index(eeg_label)
    emg_idx = labels.index(emg_label)

    eeg = f.readSignal(eeg_idx)
    emg = f.readSignal(emg_idx)

    f.close()

    # --- Determine sampling rate (can differ per channel)
    fs_eeg = sfreqs[eeg_idx]
    fs_emg = sfreqs[emg_idx]
    fs = float(np.mean([fs_eeg, fs_emg]))  # approximate common fs

    # --- Create time vector
    n_samples = min(len(eeg), len(emg))
    t = np.arange(n_samples) / fs

    df = pd.DataFrame({
        "time": t,
        "eeg": eeg[:n_samples],
        "emg": emg[:n_samples],
    })

    return df

def to_epoch(x: np.ndarray, epoch_len: int) -> np.ndarray:
    """
    Convert 1D array x into 2D array of shape (n_epochs, epoch_len).
    If len(x) is not a multiple of epoch_len, the last epoch is truncated.
    """
    n_epochs = len(x) // epoch_len
    return x[:n_epochs * epoch_len].reshape((n_epochs, epoch_len))

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

def extract_features_per_epoch(df: pd.DataFrame, cfg: Config) -> Tuple[List[EpochFeatures], dict]:
    seconds_per_epoc = int(round(cfg.epoch_sec * cfg.fs))
    eeg = to_epoch(df["eeg"].to_numpy(dtype=float), seconds_per_epoc)
    emg = to_epoch(df["emg"].to_numpy(dtype=float), seconds_per_epoc)
    n_epochs = min(eeg.shape[0], emg.shape[0])

    feats: List[EpochFeatures] = []
    for i in range(n_epochs):
        start_time = i * cfg.epoch_sec
        end_time = (i + 1) * cfg.epoch_sec

        # EMG RMS
        emg_rms = np.sqrt(np.mean(emg[i]**2))
        emg_rms = max(emg_rms, cfg.emg_rms_floor)

        # EEG Power Bands
        freqs = np.fft.rfftfreq(eeg.shape[1], d=1/cfg.fs)
        psd = np.abs(np.fft.rfft(eeg[i]))**2

        delta_power = np.sum(psd[(freqs >= cfg.eeg_band_delta[0]) & (freqs <= cfg.eeg_band_delta[1])])
        theta_power = np.sum(psd[(freqs >= cfg.eeg_band_theta[0]) & (freqs <= cfg.eeg_band_theta[1])])

        td_ratio = theta_power / (delta_power + 1e-12)  # avoid div by zero

        feat = EpochFeatures(
            start_time=start_time,
            end_time=end_time,
            td_ratio=td_ratio,
            delta_power=delta_power,
            theta_power=theta_power,
            emg_rms=emg_rms
        )
        feats.append(feat)
    # z-score features across the session
    td_arr = np.array([f.td_ratio for f in feats])
    delta_arr = np.array([f.delta_power for f in feats])
    emg_arr = np.array([max(f.emg_rms, cfg.emg_rms_floor) for f in feats])  # floor to avoid degenerate zeros

    td_z = zscore(np.log10(td_arr))            # ratios → log then z
    delta_z = zscore(np.log10(delta_arr))      # powers → log then z
    emg_z = zscore(np.log10(emg_arr))          # EMG RMS → log then z

    max_streaks = {SleepStage.REM: 0, SleepStage.NREM: 0, SleepStage.AWAKE: 0}

    current_streak = 0 
    previous_stage = None

    for i, f in enumerate(feats):
        f.td_ratio_z = float(td_z[i])
        f.delta_z = float(delta_z[i])
        f.emg_z = float(emg_z[i])
        f.stage = classify_epoch(f, cfg)

        if f.stage == previous_stage:
            current_streak += 1
        else:
            # close the previous run
            if previous_stage is not None:
                max_streaks[previous_stage] = max(max_streaks[previous_stage], current_streak)
            # start a new run
            previous_stage = f.stage
            current_streak = 1

        # keep max for the current stage up to date
        if current_streak > max_streaks[f.stage]:
            max_streaks[f.stage] = current_streak

    # finalize the last run
    if previous_stage is not None:
        max_streaks[previous_stage] = max(max_streaks[previous_stage], current_streak)

    return feats, dict(max_streaks)


def plot_hypnogram(feats: List[EpochFeatures], out_png: Path) -> str:
    t = [f.start_time for f in feats]
    y = [f.stage.value for f in feats]  # 0,1,2
    fig = plt.figure(figsize=(10, 3))
    plt.step(t, y, where="post")
    plt.yticks([0,1,2], ["AWAKE","NREM","REM"])
    plt.xlabel("Time (s)")
    plt.ylabel("Stage")
    plt.title("Hypnogram")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return fig_to_data_url(fig)

def plot_feature_dataurl(feats: List[EpochFeatures], attr: str, ylabel: str, title: str) -> str:
    t = [f.start_time for f in feats]
    v = [getattr(f, attr) for f in feats]
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t, v)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return fig_to_data_url(fig)
    
