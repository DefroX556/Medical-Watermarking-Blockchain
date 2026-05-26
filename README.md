# Medical Image Watermarking with DBSCAN & Blockchain

A production-grade system for securing medical images using **DBSCAN-guided invisible watermarking** and **blockchain-based integrity verification**.

## Features

| Feature | Description |
|---|---|
| **DBSCAN Pixel Selection** | Density-based clustering for stable watermark positions |
| **Adaptive DBSCAN** | Auto-tuning eps/MinPts per image modality (MRI/CT/X-ray/Ultrasound) |
| **AI-Powered Placement** | Ollama Cloud GLM-5.1 for intelligent embedding region selection |
| **Blockchain Ledger** | SHA-256 hash chain with Merkle tree verification |
| **9 Attack Tests** | Compression, noise, crop, rotation, scaling, histogram eq, CLAHE, median filter, watermark removal |
| **Clinical Integration** | FHIR R4 compliant ImagingStudy/DiagnosticReport generation |
| **HIPAA/GDPR Compliance** | PHI detection, AES-256-GCM encryption, RBAC audit logging |
| **Edge/Mobile Mode** | Downsampled processing with BLAKE2b fast hashing |

## Quick Start

```bash
# Clone
git clone https://github.com/DefroX556/Medical-Watermarking-Blockchain.git
cd Medical-Watermarking-Blockchain

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key (for AI mode)
cp .env.example .env
# Edit .env and add your Ollama API key

# Run
python run_demo.py                    # Standard mode
python run_demo.py --adaptive         # Adaptive DBSCAN
python run_demo.py --ai-mode          # AI-powered placement
python evaluate.py                    # Full 9-attack evaluation
```

## Project Structure

```
├── config.py               # Centralized configuration
├── run_demo.py             # Single-image demo (--adaptive, --ai-mode)
├── evaluate.py             # 9-attack robustness evaluation
├── utils.py                # Shared utilities
├── core/
│   ├── embed.py            # Watermark embedding
│   ├── extract.py          # Watermark extraction
│   ├── ai_analyzer.py      # Ollama Cloud GLM integration
│   ├── texture_analyzer.py # Gabor/GLCM texture analysis
│   ├── attack_detector.py  # GAN/manipulation detection
│   ├── batch.py            # Parallel batch processing
│   └── edge_mode.py        # Mobile/edge optimization
├── dbscan/
│   ├── dbscan_cluster.py   # Core DBSCAN pixel selection
│   └── adaptive.py         # Adaptive parameter tuning
├── blockchain/
│   ├── blockchain.py       # Hash chain ledger
│   └── merkle.py           # Merkle tree verification
├── clinical/
│   ├── interface.py        # Clinical pipeline
│   └── fhir_bridge.py      # FHIR R4 integration
├── compliance/
│   ├── audit.py            # RBAC audit logging
│   ├── encryption.py       # AES-256-GCM encryption
│   └── hipaa.py            # HIPAA/GDPR compliance
└── dataset/                # Medical images
```

## Results

| Image | PSNR (dB) | SSIM | Attack Detection |
|---|---|---|---|
| CT Scan | 28.11 | 0.9933 | 9/9 (100%) |
| MRI | 32.54 | 0.9967 | 9/9 (100%) |
| X-Ray | 29.55 | 0.9907 | 9/9 (100%) |

## Security

- API keys are stored in `.env` (gitignored) — never in source code
- AES-256-GCM encryption for metadata at rest
- Hash-chained audit logs with RBAC
- HIPAA-compliant PHI detection

## Contributors

- **Shibam Maity** ([@DefroX556](https://github.com/DefroX556))
- **Bratyabasu Mondal** ([@bratyabasu07](https://github.com/bratyabasu07))

## License

MIT
