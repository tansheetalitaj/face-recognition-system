# Face Recognition, Head Pose, Gaze, and Blink Liveness

A local webcam prototype that recognizes enrolled people with dlib-based face
embeddings and adds head-pose, iris-gaze, and blink-liveness observations.

## Current capabilities

- Load and validate one or more enrollment photographs per identity
- Cache unchanged face encodings for faster startup
- Detect and identify multiple faces from a webcam
- Reject candidates outside a configurable face-distance tolerance
- Mark both eye regions using tilt-aware facial landmarks, with Haar detection as a fallback
- Resize and skip recognition frames to reduce CPU usage
- Associate faces between processed frames and require stable identity matches
- Require an open-closed-open blink challenge before displaying `[LIVE]`
- Estimate head pitch, yaw, and roll with a six-point PnP model
- Estimate image-relative gaze from both MediaPipe iris centres with confidence gating
- Keep passive anti-spoofing disabled by default while it awaits local calibration
- Provide opt-in coarse age bands with explicit uncertainty
- Track face motion between recognition frames with optical flow
- Evaluate accuracy, false accepts, and false rejections on a labeled dataset
- Keep event logging off by default and require explicit consent when enabled
- Inventory and delete local biometric data with guarded commands
- Encrypt embeddings with Windows user-bound DPAPI by default
- Support explicit DPAPI encryption of enrollment photographs
- Track consent purpose and retention deadlines per identity
- Verify model SHA-384 and exact dependency versions
- Configure camera and recognition settings through CLI options or environment variables
- Exit with `Q` or `ESC` and always release the camera cleanly

Emotion and gender classification are intentionally not implemented: neither is
required for recognition or blink liveness, and both introduce avoidable
sensitive inferences. This remains a research prototype, not biometric authentication.

## Requirements

- Windows, Linux, or macOS
- Python 3.10 (the version currently standardized for this project)
- A webcam
- Build tools compatible with `dlib` if a prebuilt wheel is unavailable

The `face-recognition` package does not officially support Windows, although it
can work when its `dlib` dependency is installed correctly.

## Setup on Windows

From PowerShell in this directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_models.py --include-age
python verify_setup.py
```

If PowerShell blocks activation, use the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Enrollment

For better coverage, create one folder per identity and add several clear
photographs with different poses and lighting:

```text
known_faces/
  alice/
    front.jpg
    left-angle.jpg
    right-angle.png
  bob/
    front.jpg
```

Legacy single images such as `known_faces/alice.jpg` remain supported; their
filename becomes the identity. In a person folder, the folder name becomes the
identity and every valid image is another reference sample.

Each file must contain exactly one detectable face. Supported extensions are
`.jpg`, `.jpeg`, and `.png`, case-insensitively. The application refuses to
start and reports all invalid enrollment files instead of crashing or silently
choosing one face from a group photograph.

Enrollment photos are biometric personal data. Obtain consent, restrict access,
and define retention and deletion rules before enrolling other people. Do not
commit private enrollment images to a public repository.

### Encoding cache

Validated embeddings are stored in `.cache/face_encodings.enc`. On later
starts, only new or content-changed images are encoded. The cache uses JSON,
not executable pickle data, and is encrypted with Windows DPAPI for the current
user. It is excluded from Git. Copying it to another Windows account or machine
does not make it decryptable.

Force a complete rebuild after changing recognition models or troubleshooting:

```powershell
python main.py --rebuild-cache
```

For non-sensitive debugging, encryption can be explicitly disabled with
`--disable-cache-encryption`. Do not use that option with real biometric data.

### Enrollment encryption

Plaintext images are supported for initial setup. To replace one identity's
images with Windows user-bound DPAPI files, run:

```powershell
python manage_data.py encrypt-identity alice --confirm alice
```

Every encrypted file is decrypted and byte-compared before its plaintext source
is removed. This operation is not portable and is intentionally never automatic.

## Run

```powershell
python main.py
```

The original command remains available:

```powershell
python face.py
```

Useful options:

```powershell
python main.py --camera 1 --tolerance 0.55 --frame-scale 0.5 --process-every 2
python main.py --confirmations 3 --rebuild-cache
python main.py --blink-threshold 0.18
python main.py --known-faces C:\path\to\enrollment-images
python main.py --help
```

A lower tolerance is stricter and can reduce false matches at the cost of more
false rejections. Do not select a production threshold without evaluating it on
representative data.

## Configuration

CLI options override environment variables.

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `FACE_CAMERA_INDEX` | `0` | Webcam device index |
| `FACE_TOLERANCE` | `0.60` | Maximum accepted face distance |
| `FACE_FRAME_SCALE` | `0.50` | Scale used for recognition frames |
| `FACE_PROCESS_EVERY_N_FRAMES` | `2` | Recognition interval |
| `FACE_CONFIRMATION_FRAMES` | `3` | Consecutive matches required for a name |
| `FACE_BLINK_THRESHOLD` | `0.18` | Eye ratio below which eyes count as closed |
| `FACE_KNOWN_FACES_DIR` | `known_faces/` | Enrollment directory |
| `FACE_ENCODING_CACHE` | `.cache/face_encodings.enc` | DPAPI-encrypted embedding cache |

`.env.example` documents these settings. The application reads real process
environment variables; it does not automatically load a `.env` file.

## Blink liveness challenge

After identity confirmation, the label guides the participant through:

```text
Look at camera -> Blink now -> Open eyes -> [LIVE]
```

The check uses the eye aspect ratio from six facial-landmark points per eye.
If normal open eyes are repeatedly treated as closed, lower the threshold
slightly; if closed eyes are treated as open, raise it slightly:

```powershell
python main.py --blink-threshold 0.16
```

The challenge can be disabled for debugging with `--disable-liveness`. A blink
check blocks a static photograph but can still be defeated by replayed video or
more sophisticated presentation attacks. `[LIVE]` is therefore not proof of
identity or suitability for access control.

## Head pose and gaze

Head pose is estimated from six dense face landmarks with OpenCV `solvePnP` and
shown as pitch (`P`), yaw (`Y`), and roll (`R`) in degrees. These values are
approximate because the camera is not individually calibrated and a generic 3D
face is used.

Gaze uses MediaPipe's refined iris landmarks for both eyes. It reports `CENTER`,
`SCREEN LEFT`, or `SCREEN RIGHT`, optionally combined with `UP` or `DOWN`, plus
a confidence percentage. `SCREEN LEFT/RIGHT` always describes the displayed
image, so it remains unambiguous on mirrored or unmirrored cameras. The app
reports gaze as unavailable when the eyelids are closed or the two eyes disagree.
This estimates eye orientation, not attention or intent.

## Optional age estimation

Age inference is off by default. It uses a separate age-only model and never
computes gender. Download its checksum-pinned files and enable it explicitly:

```powershell
python setup_models.py --include-age
python main.py --enable-age-estimation
```

The overlay shows only a broad training-model band such as `38-43`, its maximum
class probability, and `very uncertain`, `uncertain`, or `moderate confidence`.
The bands have gaps and the estimate can be biased by image quality and training
data; it must not be treated as a person's verified age or used for eligibility.

## Passive anti-spoofing

Passive anti-spoofing is disabled by default. When explicitly enabled, the local
`anti-spoof-mn3` ONNX model must produce two
consecutive real scores at or above `0.70`. The label shows `PASSIVE CHECK` while
collecting them and `SPOOF WARNING` when the score is below threshold.

```powershell
python main.py --enable-passive-anti-spoof --passive-threshold 0.70
```

The model is downloaded only by `setup_models.py`, recorded in
`models/manifest.json`, and checked by exact size and SHA-384 on every startup.
It uses RGB input and was trained on CelebA-Spoof; it can perform differently on
other cameras, lighting, skin tones, glasses, displays, prints, and replay media.
Calibrate it with consented real and attack samples before relying on it.

## Optional consented event logging

No recognition history is stored by default. To enable metadata-only JSONL
logging, both a destination and explicit consent confirmation are required:

```powershell
python main.py --event-log logs\recognition.jsonl --consent-to-event-logging
python main.py --event-log logs\recognition.jsonl --consent-to-event-logging --event-retention-days 14
```

Records contain UTC time, identity, distance, liveness result, and track ID.
Frames are never written. Repeated events for the same identity are suppressed
for 30 seconds. Ensure every affected person has consented and apply an
appropriate retention period. Records older than 30 days are removed by default
when logging starts; change the period with `--event-retention-days`.

## Biometric data management

Inspect local enrollment and cache state:

```powershell
python manage_data.py status
```

Deletion requires exact typed confirmation:

```powershell
python manage_data.py delete-identity alice --confirm alice
python manage_data.py purge-cache --confirm PURGE
```

Deleting an identity removes only its enrolled image files and invalidates the
generated encoding cache. These actions are irreversible; back up intentionally
before using them.

Record consent and retention only after obtaining it from the participant:

```powershell
python manage_data.py record-consent alice --purpose "local research" --retention-days 90 --confirm alice
python main.py --require-consent-manifest
```

Normal mode warns about missing or expired records. Strict mode refuses to start.

## Accuracy evaluation

Create a separate dataset. Known-person folder names must exactly match the
enrollment identities. Put people who are not enrolled in `_unknown`:

```text
evaluation/
  alice/
    different-photo-1.jpg
    different-photo-2.jpg
  bob/
    different-photo.jpg
  _unknown/
    stranger-1.jpg
    stranger-2.jpg
```

Do not reuse enrollment photographs for evaluation; that produces misleadingly
optimistic results. Run:

```powershell
python evaluate.py --dataset evaluation
python evaluate.py --dataset evaluation --tolerance 0.50
python evaluate.py --dataset evaluation --sweep
```

The report includes each prediction plus identification accuracy,
false-accept rate, false-reject rate, and misidentifications. Use a dataset that
represents expected lighting, poses, glasses, cameras, and participants before
selecting a tolerance.

## Tests

The unit tests do not need a webcam or native computer-vision dependencies:

```powershell
py -3.10 -m unittest discover -s tests -v
```

They cover configuration, multi-photo enrollment, cache invalidation, eye
filtering, recognition decisions, temporal confirmation, and evaluation
metrics. Webcam integration still requires a manual test on the target computer.

## Project structure

```text
face-recognition-system/
|-- main.py                 # CLI entry point
|-- evaluate.py             # Offline accuracy evaluation
|-- manage_data.py          # Guarded biometric-data management
|-- setup_models.py         # Pinned model downloader
|-- verify_setup.py         # Environment and artifact integrity
|-- requirements-lock.txt   # Exact validated environment
|-- face.py                 # Backward-compatible launcher
|-- src/
|   |-- camera.py           # Webcam loop and visualization
|   |-- face_analysis.py    # Head-pose and iris-gaze estimation
|   |-- age_estimation.py   # Optional age-only broad-band estimate
|   |-- config.py           # Configuration and validation
|   |-- enrollment.py       # Enrollment image validation
|   |-- encoding_cache.py   # Versioned embedding cache
|   |-- anti_spoof.py       # Verified passive liveness model
|   |-- consent.py          # Consent and retention manifest
|   |-- data_protection.py  # Windows DPAPI encryption
|   |-- evaluation.py       # Accuracy and error metrics
|   |-- event_log.py        # Consent-gated metadata logging
|   |-- eye_detection.py    # Filters noisy eye detections
|   |-- liveness.py         # Blink challenge and eye ratio
|   |-- optical_flow.py     # Frame-to-frame motion tracking
|   |-- recognizer.py       # Distance-based identity matching
|   `-- tracking.py         # Face association and stable labels
|-- tests/                  # Standard-library unit tests
|-- known_faces/            # Private enrollment photographs
|-- requirements.txt
`-- .env.example
```

The XML files in the repository are legacy assets. The application uses the eye
cascade distributed with the installed OpenCV package and does not currently use
the local face or smile cascades.

## Known limitations

- Blink liveness blocks a static photo but is not proof against replay attacks.
- The passive model is camera- and domain-sensitive and needs local evaluation.
- Head pose uses a generic face/camera model and is not individually calibrated.
- Iris gaze is image-relative, approximate, and sensitive to glasses, occlusion, and lighting.
- Age output is a coarse, potentially biased estimate with explicit uncertainty.
- DPAPI-encrypted data is tied to the current Windows user and is not portable.
- Recognition accuracy varies with lighting, pose, image quality, age, and population.
- Face association starts from bounding-box overlap; optical flow projects tracks between updates.
- Three matching processed frames are required by default; this adds a short delay.
- Gaze direction does not establish attention, comprehension, or intent.
- This prototype must not be treated as proof of identity.

## Planned next milestone

- Build a guided enrollment and consent user interface
- Collect a consented local real/spoof evaluation set
- Calibrate recognition and anti-spoof thresholds from measured error rates
- Add per-camera head-pose/gaze calibration and temporal smoothing
- Add recovery/export workflows for intentionally portable encrypted backups
- Package the application as a signed desktop release

## Author

Tansheet Ali  
Independent Researcher / PhD Applicant in Artificial Intelligence and Computer Vision  
GitHub: <https://github.com/tansheetalitaj>

## License

MIT License. See `LICENSE`.
