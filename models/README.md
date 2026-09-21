# Model artifacts

`anti-spoof-mn3.onnx` is intentionally not tracked by Git. Download it with:

```powershell
python setup_models.py
```

The downloader accepts only the artifact described in `manifest.json` and
verifies its exact byte size and SHA-384 before installation. The model is the
Intel Open Model Zoo `anti-spoof-mn3` public model, based on MobileNetV3 and
trained on CelebA-Spoof. Upstream documents class 0 as real and class 1 as spoof.
The original model is distributed under the MIT License.

Model output is a risk signal, not proof of liveness. Accuracy must be measured
on the actual camera, environment, participants, and expected attack media.

The optional age-only artifacts can be installed with:

```powershell
python setup_models.py --include-age
```

They are pinned by byte size and SHA-384 in `manifest.json`. The model outputs
eight coarse age bands. It does not run by default, does not compute gender, and
must be treated as an uncertain research estimate rather than verified age.
