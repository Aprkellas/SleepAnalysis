# 💤 SleepAnalysis

A Python application for analyzing sleep state transitions (Wake / NREM / REM) from ASCII EEG and EMG datasets — originally designed for rodent sleep research.

---

## 🧠 Overview

This tool ingests raw ASCII data containing **EEG (Hz)** and **EMG (mV)** measurements over time, splits the signal into epochs, calculates spectral ratios, and classifies each epoch into a sleep state using configurable z-score thresholds.

It’s built to handle thousands of datapoints from ASCII logs or exported acquisition systems.

---

## 📂 Example Data Format

A typical ASCII file looks like this:

