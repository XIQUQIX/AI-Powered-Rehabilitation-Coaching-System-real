# AI-Powered-Rehabilitation-Coaching-System

Project Banner

## 🎯 The Problem

Recovering from an injury or surgery often requires patients to diligently perform prescribed rehabilitation exercises at home. However, two major challenges persist:

- **Incorrect form** — Without real-time professional guidance, many patients perform exercises improperly, which can slow recovery, worsen the injury, or lead to new complications.
- **Poor adherence** — Patient compliance (actually completing the full prescribed routine and frequency) remains one of the biggest barriers to successful at-home rehabilitation.

Traditional in-person physiotherapy is expensive, time-consuming, and not always accessible — especially in rural areas or during long-term recovery.

## 🚀 Our Solution

**Virtual Physiotherapy Assistant (VPA)** is an intelligent AI system that acts as your personal virtual physiotherapist — available anytime, anywhere, directly from your phone or webcam.

### Core Capabilities

- **Real-time pose estimation & movement analysis** — Uses your camera to track body keypoints and evaluate exercise execution.
- **Detailed, constructive feedback** — Tells you exactly what you're doing **correctly**, **moderately well**, or **poorly**, with specific, actionable suggestions to correct form (e.g. "Keep your knee aligned over your ankle — try shifting weight slightly forward").
- **Retrieval-Augmented Generation (RAG)** recommendation engine — Personalizes advice based on:
  - Your specific injury / condition
  - Doctor / physiotherapist recommendations
  - Evidence-based rehab protocols for common injuries
- **Patient-centric design** — Aims to increase adherence through clear, encouraging, human-like coaching.

The goal is simple: help people recover **faster**, **safer**, and **more consistently** from home — while reducing the burden on healthcare systems.

## ✨ Key Features (Initial Version)

- Video-based real-time exercise assessment
- Multi-level feedback (good / moderate / needs improvement)
- Personalized recommendations via RAG (injury-specific + protocol-aware)
- Chat interface for asking questions about exercises, pain, or progress
- (Planned) Progress tracking & adherence reports

## 🛠️ Technology Highlights

- **Computer Vision** → Human pose estimation (likely MediaPipe / OpenPose / RTMPose family)
- **AI Feedback Engine** → LLM-powered critique + natural language generation
- **Retrieval-Augmented Generation (RAG)** → For retrieving and grounding advice in trusted physiotherapy knowledge
- **Frontend** → (Web / mobile app — webcam access)
- **Backend** → Python-based inference pipeline

## 📦 Environment Setup

### Prerequisites
- **Conda** (Miniconda or Anaconda) installed on your system
- **Python 3.11** (specified in the environment file)

### Installation Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/DS5500-Team-11-AI-Powered-Rehab/AI-Powered-Rehabilitation-Coaching-System.git
   cd AI-Powered-Rehabilitation-Coaching-System
   ```

2. **Create the Conda environment** from the provided environment file:
   ```bash
   conda env create -f rehab_ai_env.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate rehab_ai_env
   ```

### What's Included

The environment includes:
- **Core scientific stack**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch (CPU-only), TorchVision, TorchAudio
- **RAG / Vector Database**: ChromaDB, Sentence Transformers
- **LLM Frameworks**: LangChain, LangGraph
- **LLM Clients**: OpenAI, Anthropic, Ollama
- **Data Processing**: PyPDF, python-docx, BeautifulSoup4
- **Jupyter**: Notebook environment for development and experimentation
- **Additional tools**: Transformers, Accelerate, Spacy, and more

### Deactivating the Environment

When you're done, deactivate the environment:
```bash
conda deactivate
```

## 📁 Project Structure

```
AI-Powered-Rehabilitation-Coaching-System/
│
├── README.md                        # This file — system overview
├── LICENSE                          # Project license
├── rehab_ai_env.yml                 # Conda environment specification
├── .env / .env.example              # Environment variables (API keys, model configs)
├── .gitignore                       # Git ignore rules
│
├── cache/                           # Cached responses & precomputed data
│   └── tier1_responses/             # Tier 1 cached coaching responses
│
├── figures/                         # Project visualizations & diagrams
│
├── notebooks/                       # Jupyter notebooks for exploration & evaluation
│   ├── llm_comprehensive_evaluation.ipynb
│   ├── validated_test_questions.json
│   └── evaluation_results/
│       ├── aggregate_metrics.csv
│       ├── EVALUATION_SUMMARY.md
│       └── *.csv                    # Detailed model evaluation results
│
├── src/                             # Production code
│   │
│   ├── cv/                          # Computer Vision pipeline
│   │   ├── __init__.py
│   │   ├── extract_pose_cache.py    # Extract pose data from video cache
│   │   ├── infer_stream.py          # Real-time pose inference on video stream
│   │   ├── precompute_memmap.py     # Precompute pose data to memory-mapped files
│   │   └── train_from_memmap.py     # Train models from precomputed pose data
│   │
│   ├── integration/                 # Integration layer (CV → LLM bridge)
│   │   ├── __init__.py
│   │   ├── integration_layer.py     # Core temporal filtering & routing logic
│   │   ├── graph.py                 # LangGraph workflow integration
│   │   ├── state.py                 # State management for integration layer
│   │   ├── main.py                  # Entry point for integration pipeline
│   │   └── README.md                # Integration layer documentation
│   │
│   ├── rag/                         # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── ingest.py                # Chunk & embed PT guidelines → ChromaDB
│   │   ├── retriever.py             # Query interface over ChromaDB
│   │   └── prompt_templates.py      # Tier 2 slot-based prompts
│   │
│   ├── agents/                      # LangGraph multi-agent system
│   │   ├── __init__.py
│   │   ├── state.py                 # Shared LangGraph state schema
│   │   ├── movement_analysis.py     # Movement Analysis Agent
│   │   ├── coaching.py              # Coaching Agent (conversational memory)
│   │   ├── progress.py              # Progress Tracking Agent
│   │   └── orchestrator.py          # LangGraph graph definition & routing
│   │
│   ├── feedback/                    # Feedback generation & delivery
│   │   ├── __init__.py
│   │   ├── tier1_cache.py           # Load/serve pre-computed responses
│   │   ├── tier2_generator.py       # RAG + LLM generation
│   │   ├── tier3_reasoner.py        # Full agent reasoning pass
│   │   └── delivery.py              # Timing logic (immediate / rep-end / rest)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Load .env, model names, thresholds
│       └── logging.py               # Logging utilities
│
├── tests/                           # Testing suite
│   ├── integration_testing/
│   │   ├── test_integration.py      # Integration layer test runner
│   │   ├── test_tune_thresholds.py  # Threshold tuning analyzer
│   │   ├── test_visualize.py        # Visualization generator
│   │   ├── generate_synthetic_data.py  # Generate synthetic test data
│   │   ├── quick_start_test.sh      # Automated test pipeline
│   │   ├── TEST_README.md           # Comprehensive testing guide
│   │   └── tuned_config.json        # Validated threshold configuration
│
├── scripts/                         # One-off runnable scripts
│   ├── ingest_pt_data.py            # Populate ChromaDB with PT guidelines
│   ├── build_tier1_cache.py         # Pre-compute top mistake responses
│   └── run_demo.py                  # End-to-end demo runner
│
└── docs/
    └── api_contracts.md             # CV ↔ Integration ↔ LLM interface specs
```

## Why This Matters

Incorrect exercise performance and low adherence are well-documented causes of prolonged recovery times and increased healthcare costs. By combining state-of-the-art **pose estimation**, **generative AI**, and **personalized retrieval**, VPA aims to bring high-quality, 24/7 physiotherapy guidance to anyone with a smartphone or laptop.

We're building this as an open-source project to encourage collaboration between AI researchers, physiotherapists, clinicians, and rehab tech enthusiasts.

## 🚧 Current Status

Early / proof-of-concept stage  
Actively developing core pose → feedback loop and RAG integration

Contributions, feedback, and domain expertise (especially from physiotherapists) are **very welcome**!

---

**Topics**: #pose-estimation #human-pose-estimation #computer-vision #rehabilitation #physiotherapy #healthcare-ai #exercise-feedback #rag #ai-healthcare #physical-therapy

Star ⭐ the repo if you're interested in AI for healthcare & rehabilitation!

Let's make high-quality rehab accessible to everyone.
