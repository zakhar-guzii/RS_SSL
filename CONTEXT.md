# Ubiquitous Language — HAR Activity Recognition App

## Glossary

- **App** — cross-platform mobile app running on the user's phone. Uses the phone's own IMU (accelerometer + gyroscope) as the sensor. The user presses a button, the app records motion, and displays the predicted Activity.
- **Activity** — one of the 5 labels shared by all merged datasets: `downstairs`, `sit`, `stand`, `upstairs`, `walk`. Encoded 0–4 in alphabetical order.
- **Recording** — the ~15-second capture of raw, timestamped 6-channel motion samples — accelerometer `(ax, ay, az)` + gyroscope `(gx, gy, gz)` — triggered by the button press, sent as-is to the Backend. The App does no preprocessing. Recordings are ephemeral: the Backend classifies and discards them, never persisting user motion data.
- **Canonical Signal** — the standardized form all model input takes: 6 channels `[ax, ay, az, gx, gy, gz]` — total acceleration (gravity included) in g units plus angular velocity in rad/s — resampled to 50 Hz. Defined by the merge pipeline (`src/data_merge.py`); the Backend must transform every Recording into this form before inference.
- **Window** — a 128-sample (2.56 s) slice of Canonical Signal; the unit the model actually classifies. A Recording yields many Windows.
- **Prediction** — the Backend's answer for one Recording: the top Activity with a confidence score, derived by averaging per-Window class probabilities across the whole Recording. Also carries per-class probabilities and the per-Window label sequence.
- **Model Bundle** — one deployable trained model: weights plus everything needed to reproduce training-time preprocessing (normalization stats, label encoding, architecture config) and a human-readable name/description (e.g. whether it was SSL-pretrained). The Backend serves all available Model Bundles; the App user picks one before recording.
- **Backend** — a server process hosted on a team laptop. Stores trained model checkpoints, receives Recordings, runs inference, and returns the predicted Activity. It is a standalone service (not a Jupyter notebook); notebooks remain for experimentation only.
