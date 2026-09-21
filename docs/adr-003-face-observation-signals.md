# ADR-003: Face observation signals and sensitive-attribute boundaries

## Status

Accepted for the local prototype.

## Context

The webcam loop needs head pose, gaze, blink liveness, optional event logging,
and optional age estimation. Passive anti-spoofing is producing uncalibrated
false rejections. Emotion and gender are not required for recognition.

## Decision

- Disable passive anti-spoofing by default; retain an explicit experimental flag.
- Keep the existing open-closed-open blink challenge as the default liveness gate.
- Run one refined MediaPipe face mesh on each processed frame and associate it
  with recognition boxes by intersection-over-union.
- Derive head pose through OpenCV PnP and iris gaze from both eyes. Suppress gaze
  when eyes are closed or the two-eye signal is inconsistent.
- Keep event logging disabled and consent-gated. Log metadata, never frames.
- Keep age off by default. Use an age-only model, expose broad bands and model
  confidence, and label uncertainty explicitly.
- Do not implement emotion or gender inference without a separately documented,
  ethically reviewed research requirement.

## Data flow

```text
camera frame
  -> face recognition + blink landmarks
  -> refined face mesh -> head pose + iris gaze
  -> optional age-only network
  -> temporal tracker -> overlay
  -> optional consent-gated metadata event
```

Frames and derived pose/gaze/age observations remain in memory. The default
event log contains identity, recognition distance, liveness status, track ID,
and timestamp only.

## Trade-offs

MediaPipe adds installation size and CPU work but provides iris landmarks that
the existing 68-point face model cannot. PnP uses a generic face and approximate
camera intrinsics, so pose is suitable for UI feedback, not precision metrology.
Age bands reduce false precision but do not eliminate dataset bias. Avoiding
emotion and gender reduces sensitive inference and scope.

## Revisit when

- a calibrated camera model is available;
- consented local gaze/pose accuracy data has been collected;
- a concrete use case justifies storing derived signals;
- passive anti-spoofing has a measured local operating threshold.
