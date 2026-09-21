# ADR-001: Local Liveness and Privacy Controls

**Status:** Accepted  
**Date:** 2026-09-21  
**Decider:** Project owner

## Context

The application handles biometric photographs and embeddings. Milestone 2
improved recognition quality but did not distinguish a live participant from a
static photograph, provide deletion tooling, or control recognition event logs.
The application remains a local research prototype and must work on the current
CPU-only Windows setup without adding cloud processing.

## Decision

- Use a per-track open-closed-open blink challenge calculated from six-point
  dlib eye landmarks.
- Require stable identity confirmation before presenting the blink challenge.
- Describe this as a basic liveness signal, not spoof-proof authentication.
- Keep event logging disabled by default and require an explicit consent flag.
- Log metadata only; never write camera frames.
- Provide guarded commands to inventory and delete enrollment data and caches.
- Keep biometric images, embeddings, evaluation data, and logs out of Git.
- Use threshold sweeps on independent labeled images for local calibration.

## Options Considered

### Blink challenge

Low deployment complexity and no extra model download, but replayed video or a
sophisticated presentation attack can pass it.

### Dedicated anti-spoof neural model

Potentially stronger against print and screen attacks, but adds model supply,
licensing, hardware, calibration, and demographic-validation requirements.

### Cloud liveness API

Could provide a maintained model but sends biometric data to a third party,
introduces cost and network dependency, and conflicts with local-first privacy.

## Consequences

- Recognition now has a short deliberate challenge before `[LIVE]` appears.
- The larger landmark model uses more CPU than the previous two-point model.
- Event logging cannot be enabled accidentally with only a destination path.
- Users can delete local identity data without manually locating cache entries.
- High-security use still requires a validated anti-spoof model and threat model.

## Follow-up

1. Collect consented, representative evaluation and spoof-attempt data.
2. Tune the blink threshold and recognition tolerance from measured results.
3. Evaluate a dedicated passive anti-spoof model before security-sensitive use.
4. Add encrypted storage if enrollment data moves beyond a single-user machine.
