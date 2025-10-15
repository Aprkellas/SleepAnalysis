# app.py
from __future__ import annotations
from tempfile import NamedTemporaryFile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from analysis import run_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

ALLOWED_EXT = {".edf", ".EDF"}

def allowed_file(filename: str) -> bool:
    from pathlib import Path
    return Path(filename).suffix in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "edf" not in request.files:
        return jsonify({"error": "Missing EDF file field 'edf'"}), 400

    edf = request.files["edf"]
    if edf.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(edf.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    fs_override = request.form.get("fs", type=float)  # None or float
    epoch_sec = request.form.get("epoch_sec", type=float, default=10.0)
    z_thr_emg_awake = request.form.get("z_thr_emg_awake", type=float, default=0.5)
    z_thr_td_rem = request.form.get("z_thr_td_rem", type=float, default=0.5)
    z_thr_delta_nrem = request.form.get("z_thr_delta_nrem", type=float, default=0.5)
    eeg_label = request.form.get("eeg_label") or None
    emg_label = request.form.get("emg_label") or None

    # pyEDFlib needs a real path, so use a temp file that auto-deletes
    safe_name = secure_filename(edf.filename)
    with NamedTemporaryFile(suffix=".edf") as tmp:
        edf.save(tmp.name)
        try:
            result = run_analysis(
                edf_path=tmp.name,
                fs_override=fs_override,
                epoch_sec=epoch_sec,
                z_thr_emg_awake=z_thr_emg_awake,
                z_thr_td_rem=z_thr_td_rem,
                z_thr_delta_nrem=z_thr_delta_nrem,
                eeg_label=eeg_label,
                emg_label=emg_label,
            )
        except Exception as ex:
            return jsonify({"error": f"Analysis failed: {ex}"}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)
