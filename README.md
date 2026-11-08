# OCR Manager

**Unburn your subtitles.** Drop in a season of hardsubbed episodes; OCR Manager tunes itself to each one, reads every line and hands you timed `.ass` files.

[![OCR Manager](.github/readme/website.png)](https://ocr-manager.blyat.uk/)

**Website: [ocr-manager.blyat.uk](https://ocr-manager.blyat.uk/)**

## Download

Get the latest build from the [releases page](https://github.com/blyat-uk/ocr-manager/releases/latest):

| OS | File |
|---|---|
| Windows 10 build 18362+ / Windows 11 (x64) | `-win.exe` installer (per-user, no admin rights needed), or `-win.zip` portable |

Each release lists every file and what it is for, plus `SHA256SUMS.txt` to verify them against.

## Requirements

Windows x64 only. The OCR engine runs on **DirectML** through ONNX Runtime, so it uses any DirectX 12 capable graphics adapter (NVIDIA, AMD or Intel) and needs no CUDA toolkit, no NVIDIA driver build and no `nvidia-smi`. Without a DX12 adapter it falls back to the CPU: slower, but it still works.

**Nothing is installed on first run.** The OCR engine, its models and ffmpeg ship inside the download.

Data lives in your user data folder:

| OS | Data folder |
|---|---|
| Windows | `%LOCALAPPDATA%\ocr-manager` |

It holds `ocr-manager.log` and `models/`: the default languages read with the models bundled in the app, and a language whose script needs a different model (Cyrillic, Devanagari, Arabic, Thai, Greek, Korean, Tamil, Telugu, Georgian) downloads that one model on first use. `--download-models` fetches them all up front for offline use.

## Platform notes

- The installer is not code-signed; SmartScreen may ask you to confirm (More info → Run anyway).
- Video is decoded by the bundled ffmpeg, and every frame's timestamp is read from it rather than estimated, so subtitles land on the frame they were read from.

## Run from source

Windows and Python 3.11 or 3.12. `ffmpeg` and `ffprobe` are not something you install: the app looks for them in its own `bin/` folder, so put them there.

```powershell
git clone https://github.com/blyat-uk/ocr-manager.git
cd ocr-manager
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# ffmpeg + ffprobe go in .\bin (see below)
.venv\Scripts\python main.py [folder]
```

`bin/` is not in git (the binaries are ~110 MB each), so a fresh clone needs them once:

```powershell
$zip = "$env:TEMP\ffmpeg.zip"
Invoke-WebRequest "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-31-13-27/ffmpeg-n8.1.2-50-g1a748fe2cd-win64-gpl-8.1.zip" -OutFile $zip
Expand-Archive $zip -DestinationPath "$env:TEMP\ffmpeg"
New-Item -ItemType Directory -Force bin | Out-Null
Copy-Item "$env:TEMP\ffmpeg\ffmpeg-n8.1.2-50-g1a748fe2cd-win64-gpl-8.1\bin\ffmpeg.exe"  bin\
Copy-Item "$env:TEMP\ffmpeg\ffmpeg-n8.1.2-50-g1a748fe2cd-win64-gpl-8.1\bin\ffprobe.exe" bin\
```

Take the **gpl** build for development: the suite synthesizes test clips and needs `libx264`/`libx265`, which the LGPL build does not carry. The released app only decodes, so it ships the LGPL build. `core.ffmpeg` finds whichever copy is in `bin/`; set `OCR_MANAGER_FFMPEG_DIR` to point it somewhere else, and `PATH` is only consulted when `bin/` has nothing.

## Develop

```powershell
$env:QT_QPA_PLATFORM = "offscreen"; .venv\Scripts\python -m pytest tests\ -m "not slow"   # tests
.venv\Scripts\python tools\fidelity_check.py                                              # golden OCR output
.venv\Scripts\python packaging\build.py                                                   # release build
```

Pushing a `vX.Y.Z` tag that matches `app/version.py` builds, tests and publishes the Windows release.

## License

[MIT](LICENSE)
