# ADR-002: Layered Local Biometric Hardening

**Status:** Accepted  
**Date:** 2026-09-21  
**Decider:** Project owner

## Requirements and constraints

- Continue operating locally on the existing Windows CPU environment.
- Do not send frames, photographs, or embeddings to cloud services.
- Detect more presentation attacks than a blink challenge alone.
- Protect biometric data at rest without embedding portable keys in the repo.
- Make consent, retention, model provenance, and dependency versions auditable.
- Preserve an explicit development escape hatch without silently weakening defaults.

## Design

```text
camera frame
  -> face recognition and landmarks
  -> stable temporal identity
  -> active blink challenge
  -> passive anti-spoof ONNX score
  -> [LIVE] only when both liveness layers pass

enrollment image -> optional Windows DPAPI file
embedding cache  -> Windows DPAPI ciphertext by default
model artifact   -> pinned source + size + SHA-384 verification
identity         -> consent purpose + retention deadline
```

## Decisions

1. Use Intel Open Model Zoo `anti-spoof-mn3` through ONNX Runtime on CPU.
2. Require both active blink liveness and two passive real classifications.
3. Never download a model during normal application startup.
4. Fail closed on missing or checksum-invalid models.
5. Use Windows DPAPI, binding encrypted biometric data to the current user.
6. Keep enrollment conversion explicit because it replaces plaintext originals.
7. Use Lucas-Kanade optical flow between expensive recognition frames.
8. Keep consent assertions user-driven; software must not invent consent records.
9. Lock the exact validated Windows/Python package environment.

## Trade-offs and consequences

- The passive model adds CPU latency and can reject genuine users when the camera
  domain differs from its training data.
- RGB passive anti-spoofing still cannot guarantee resistance to unseen attacks.
- DPAPI files are intentionally not portable to another user or machine.
- Plaintext enrollment remains supported for migration and cross-platform use;
  strict deployments should encrypt it and require the consent manifest.
- Optical flow improves box motion but does not replace identity recognition.

## Revisit when scaling

- Evaluate depth or infrared sensors for higher-assurance liveness.
- Benchmark anti-spoof error rates on representative print, screen, and replay attacks.
- Introduce managed key storage for multi-user or server deployments.
- Replace local files with an access-controlled biometric vault if deployment expands.
