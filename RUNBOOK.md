# Install and reproduce the frozen entry

Use Windows x64, Git and uv. Obtain the final clean public repository and record its actual
public commit. Do not substitute a private research URL. From this source directory:

```powershell
uv venv --python 3.12.10 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-plog.lock
.venv/Scripts/python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_competition.ps1 -PreflightOnly
```

Require tests ending OK and Preflight passed. The source-only preflight opens no competition
path and checks locked versions,17 canonical source hashes and exact config/model bytes.
The included tests cover generated199319-frame selected inference, official validation,
independent regeneration, actual byte equality, input failures and preservation of existing
synthetic outputs. They can take several minutes. No GT payload or external corpus is needed.

On the licensed machine, close hosted tooling. Keep inputs/outputs outside this source checkout.
For an already prepared local bundle, in a normal local PowerShell terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_competition.ps1 -DataDir 'D:\PLOG-input' -OutputDirectory 'D:\PLOG-output-v1'
```

Replace the two quoted local paths. The selected wrapper reads tactical ATD only, chooses the
exact frozen config/weights, generates199319 data rows and runs the official validator twice
with independent byte regeneration. It refuses existing outputs; choose a new output directory.
No network, retraining, event option, target adaptation or challenger selection is required.
For original download ZIPs, FIRST CONTACT safely prepares exact assets and calls this same runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_first_contact.ps1 -SourceDirectory 'D:\PLOG-downloads' -DestinationDirectory 'D:\PLOG-prepared'
```

Videos are not assembled. Earlier outputs are preserved in numbered directories. Do not edit
tracking or prediction labels. Existing operator-validated V1 files remain the fallback; no
rerun is required merely because source was exported. Keep local CSVs/manifests/logs private.
