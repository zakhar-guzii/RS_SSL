# Canonical Signal is 6-channel (accelerometer + gyroscope); WISDM is dropped

The App captures both accelerometer and gyroscope (the phone IMU exposes both), and the App developer wants the Backend to actually use the gyroscope rather than discard it. Previously the Canonical Signal was 3-channel, accelerometer-only: total acceleration `(x, y, z)` in g, 50 Hz — the one modality all four merged source datasets (UCI HAR, WISDM, MotionSense, HHAR) happened to share.

We make the Canonical Signal **6-channel** — `[ax, ay, az, gx, gy, gz]`, total acceleration in g plus angular velocity in rad/s, resampled to 50 Hz — and **drop WISDM entirely** from the merged training set. WISDM v1.1 is accelerometer-only; it has no gyroscope signal to recover, so it cannot contribute the three new channels. The merged dataset is therefore rebuilt from the three sources that do carry gyroscope: UCI HAR (`body_gyro_*`), MotionSense (`rotationRate.*`), and HHAR (`Phones_gyroscope.csv`). Every Model Bundle is retrained on this 6-channel data, and the `/predict` wire contract grows its per-sample row from `[t, x, y, z]` to `[t, ax, ay, az, gx, gy, gz]`.

Gyroscope units are `rad/s` across all three kept sources **and** on both mobile platforms (iOS CoreMotion `rotationRate`, Android `Sensor.TYPE_GYROSCOPE`), so — unlike the accelerometer — gyroscope needs no unit declaration or conversion anywhere in the pipeline.

The alternatives we rejected:
- **Keep WISDM and zero-fill its gyroscope.** WISDM was the single largest source (~46% of the old merge). Zero-filling would make "no rotation" a systematic artifact of a large, non-random slice of the data — the model would learn source leakage (gyro≈0 ⇒ WISDM's activity mix) rather than real motion. Scientifically dirty; rejected.
- **Accept gyroscope on the wire but ignore it (stay 3-channel internally).** Zero retraining, but it permanently wastes a signal the App already sends and the domain values (gyroscope strongly disambiguates stairs vs. walking). Rejected in favor of doing it properly.

The trade-off we accept: dropping WISDM removes the largest and most device-diverse accelerometer source, so the merged set is smaller (~60k vs ~62k windows) and leans on three sources instead of four — a modest generalization risk. In exchange, every training window is genuinely multimodal and matches what the phone actually captures. This is hard to reverse (all bundles must be retrained together to share the 6-channel input contract), which is why it is recorded here.
