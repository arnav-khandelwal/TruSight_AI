# TruSight AI

Multi-dimensional video forensics and verification system for detecting deepfakes, AI-generated synthetic content, and miscontextualized footage.

## 🎯 Overview

**TruSight AI** provides forensic analysis across three independent modules. Each fuses pretrained neural detectors with training-free signal-processing cues, and every score is evaluated on real public data:
- **Deepfake Detection** - OpenCV YuNet face detection + a pretrained face-forgery classifier, fused with a frequency-domain artifact cue, plus temporal flicker coherence across frames
- **AI-Generated Detection** - an ensemble of two pretrained AI-vs-real classifiers fused with frequency-domain and Error-Level-Analysis cues
- **Context Integrity** - CLIP zero-shot scene-vs-claim matching and continuous claim alignment, CLIP-embedding temporal reuse against a configurable reference set (with an honest "no reference database" state), and lighting/time consistency

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- FFmpeg (for video frame decoding)

### Backend Setup

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt      # installs CPU PyTorch by default
python app.py
```

Backend runs at `http://localhost:8000`.

**First run** downloads the pretrained models (~1.5 GB total: the deepfake
classifier, the AI-image classifier, and CLIP) and caches them, so the first
startup takes a few minutes. Subsequent starts take ~20 seconds. The YuNet face
detector (`backend/models/face_yunet.onnx`) is bundled.

**Optional GPU (NVIDIA):** for faster inference, install the CUDA build of
PyTorch instead (the code auto-detects and uses the GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Frontend Setup

```bash
npm install
npm run dev
```

Frontend runs at `http://localhost:3000` (or `http://localhost:5173` via Vite).

## ☁️ Deployment

The models need ~2-3 GB RAM. The recommended free setup is **backend on Hugging Face Spaces** (free CPU tier, 16 GB RAM) and **frontend on Vercel**.

### Backend → Hugging Face Spaces (Docker)
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space): **SDK = Docker**, hardware = **CPU basic (free)**.
2. Push the contents of the `backend/` folder to the Space's git repo (it contains a `Dockerfile` and a Space `README.md`). For example:
   ```bash
   cd backend
   git init && git add . && git commit -m "TruSight AI backend"
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
3. The Space builds the image (models are baked in, so it's ready on boot) and serves at `https://<user>-<space-name>.hf.space`. Check `/health`.

### Frontend → Vercel
1. Import your repository at [vercel.com/new](https://vercel.com/new). Framework auto-detects **Vite**.
2. Add an environment variable **`VITE_API_URL`** = your Space URL (e.g. `https://<user>-<space-name>.hf.space`).
3. Deploy. The CORS config already allows any `*.vercel.app` origin.

## 📁 Project Structure

```
trusight-ai/
├── backend/
│   ├── app.py                      # FastAPI main application
│   ├── config.py                   # weights, bands, model IDs (single source of truth)
│   ├── forensic_signals.py         # training-free frequency + ELA cues
│   ├── clip_backbone.py            # shared CLIP (reuse embeddings + zero-shot)
│   ├── analyze_deepfake.py         # YuNet faces + forgery classifier + frequency + flicker
│   ├── analyze_synthetic.py        # AI-detector ensemble + frequency + ELA
│   ├── analyze_context.py          # Context integrity (scene / reuse / lighting)
│   ├── temporal_analysis.py        # Reuse detection (CLIP embeddings) + intra-clip dedup
│   ├── contextual_analysis.py      # CLIP scene-vs-claim + claim alignment
│   ├── environmental_analysis.py   # Lighting/time consistency
│   ├── tests/                      # pytest unit tests (no model download)
│   ├── models/                     # face_yunet.onnx (bundled)
│   └── requirements.txt            # Python dependencies
├── eval/                           # honest evaluation on real public data
│   ├── prepare_data.py             # download labelled real-vs-fake subsets
│   ├── evaluate.py                 # AUC, calibration, latency, robustness, ablation
│   ├── make_figures.py             # ROC / AUC / calibration / robustness figures
│   └── results/                    # metrics JSON + figures
├── index.tsx / index.html          # React frontend (single-file, minimalist beige & brown theme)
└── package.json                    # Node dependencies
```

## 🔬 Technical Architecture

Fusion weights, decision bands, and model identifiers are centralized in
[`backend/config.py`](backend/config.py); the training-free cues live in
[`backend/forensic_signals.py`](backend/forensic_signals.py).

### Module 01: Deepfake Detection
- **Models**: OpenCV YuNet (face detection) + `dima806/deepfake_vs_real_image_detection` (ViT)
- **Method**: detect up to several faces per frame, crop, and fuse the forgery classifier with a frequency-domain artifact score per face; measure temporal flicker (frame-to-frame score instability) as an added cue. No-face videos return LOW.
- **Output**: Manipulation likelihood (0-100) + temporal coherence

### Module 02: AI-Generated Detection
- **Models**: ensemble of `umm-maybe/AI-image-detector` and `Organika/sdxl-detector`
- **Method**: average the ensemble's "artificial" probability per frame, fused with frequency-domain and Error-Level-Analysis cues; aggregate across frames with per-frame agreement.
- **Output**: Synthetic-media likelihood (0-100)

### Module 03: Context Integrity
- **Model**: a single shared CLIP-ViT-Base/32 backbone (used for both reuse embeddings and scene zero-shot)
- **Method**: scene-vs-claim category match plus a continuous claim-alignment score (CLIP); temporal reuse via CLIP-embedding similarity against a configurable reference database, with an intra-clip near-duplication check and an honest "no reference database" state; lighting/time consistency.
- **Output**: Context-integrity risk score (0-100)

## 🎨 Frontend Features

- **Three Independent Analysis Sections** - Separate UI for each forensic module
- **Minimalist Beige & Warm Brown Theme** - Clean, refined, editorial aesthetic designed for readability and focus
- **Real-time Processing** - Upload video and receive results in < 60 seconds
- **Explainable Results** - Detailed forensic evidence for each module
- **Responsive Architecture** - Optimized for desktop and mobile workflows

## 📡 API Usage

### Analyze Video

```bash
POST http://localhost:8000/analyze/deepfake
POST http://localhost:8000/analyze/synthetic
POST http://localhost:8000/analyze/context

Content-Type: multipart/form-data

video: <file>
claim: "Video description text" (for context module only)
```

### Response Format

```json
{
  "riskScore": 99.8,
  "riskLevel": "HIGH",
  "summary": "Detailed analysis summary",
  "signals": {
    "forensic": {
      "name": "Deepfake Manipulation Likelihood",
      "status": "fail",
      "confidence": 0.8,
      "details": "Analyzed 6 detected face(s) ..."
    },
    "temporal": {
      "name": "Face-level Detection Consistency",
      "status": "pass",
      "confidence": 0.99,
      "details": "Face-level score consistency: 99.9% ..."
    }
  }
}
```

`riskLevel` is `LOW` / `MEDIUM` / `HIGH`. Each signal's `status` is `pass` / `warn` / `fail`.

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- PyTorch + Hugging Face Transformers (+ tf-keras for the TF detectors)
- OpenCV (YuNet face detection)
- OpenAI CLIP (ViT-B/32), shared for reuse embeddings and zero-shot
- Training-free forensic cues: Fourier high-frequency ratio, spectral peakiness, Error Level Analysis
- Pretrained detectors: `dima806/deepfake_vs_real_image_detection`, `umm-maybe/AI-image-detector`, `Organika/sdxl-detector`

**Frontend:**
- React 19
- TypeScript
- TailwindCSS (Beige & Warm Brown Minimalist Palette)
- Vite
- Lucide Icons

## 🔒 Privacy & Security

- No biometric identification - only manipulation pattern detection
- Stateless processing - videos not permanently stored
- TLS encryption for data transmission
- No user data training

## 👥 Developers

- **Arnav Khandelwal** - AI Architecture & Lead Systems
- **Vighnesh Bajpai** - Full-Stack & Model Engineering
- **Rudrapratap Singh Dogra** - Forensic Signals & Backend Systems

## 📄 License

This is a technical demonstration and research project.

## 🙏 Acknowledgments

- [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection) - deepfake face classifier
- [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) and [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) - AI-generated image classifiers
- OpenAI CLIP (ViT-B/32)
- OpenCV YuNet face detector
- Evaluation data: CIFAKE, the Hemg AI-vs-real set, and OpenDeepfake-Preview (public Hugging Face datasets)
- Hugging Face Transformers and the PyTorch community
