# Multi-AI-Agent Crisis Management & Relief Coordinator

An end-to-end multi-agent NLP + computer vision system for real-time tornado detection, situational awareness, and disaster response coordination, integrating radar data, social media signals, and official FEMA guidelines.

Built to simulate how AI systems can assist emergency operators in high-stakes, time-critical disaster scenarios.

---

## Project Overview

Natural disasters like tornadoes generate fragmented, noisy, and rapidly evolving signals across radar systems, social media, and emergency protocols. Human operators must synthesize this information under extreme time pressure.

This project builds a modular, multi-agent AI pipeline that:

- Detects tornado likelihood from NEXRAD radar data
- Classifies crisis-related tweets in real time
- Retrieves authoritative FEMA response guidelines via Retrieval-Augmented Generation (RAG)
- Produces a structured, operator-ready response briefing

### The system is designed to be:
- **Explainable**
- **Scalable**
- **Grounded in verified emergency doctrine**
- **Extensible to other disaster types** (floods, wildfires, earthquakes)

---

## System Architecture (Multi-Agent Design)

```
┌──────────────┐      ┌────────────────┐
│ Radar Input  │ ──▶  │ Radar Agent     │
└──────────────┘      │ (CNN Detector)  │
                      └────────────────┘
                               │
                               ▼
┌──────────────┐      ┌────────────────┐
│ Tweet Stream │ ──▶  │ Tweet Agent     │
└──────────────┘      │ (LoRA-DistilBERT)│
                      └────────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │ Resource Coordination   │
                   │ Agent (RAG + FEMA KB)   │
                   └────────────────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │ Summarizer Decision     │
                   │ Agent (Operator Brief) │
                   └────────────────────────┘
```

---

## 🛰️ Radar Tornado Detection (Computer Vision)

### Dataset
- **TorNet (2013–2022)**
  - Multi-channel NEXRAD radar volumes
  - Severe class imbalance (≈ 6–7% confirmed tornadoes)

### Model
- **Lightweight CNN** (TinyTorCNN / SimpleCNN)
- **15-channel engineered input:**
  - Radar variables (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH)
  - Range maps, inverse-range maps
  - Quality masks

### Key Techniques
- **Focal Loss** (γ = 2.0) for rare event learning
- **Positive class oversampling** + Tomek Links cleaning
- **Time-aware cross-validation** (rolling / walk-forward)
- **Threshold optimization** using F1 score (not accuracy)

### Why Not ResNet / EfficientNet?
Radar imagery differs fundamentally from natural images — pretrained vision backbones showed no meaningful gain while adding computational cost. A custom CNN proved more stable and interpretable.

---

## Tweet Classification (NLP)

### Task
Classify tweets during tornado events into:
- `warning`
- `damage report`
- `resource request`
- `other / chatter`

### Data
- Tweets collected during major U.S. tornadoes
- Weak supervision using **Snorkel** labeling functions
- Heuristic signals (keywords, URLs, off-topic filters)

### Model
- **DistilBERT + LoRA (PEFT)**
- Separate binary (on-topic) and multi-class heads
- HuggingFace Transformers pipeline

### Why LoRA?
- ~0.1–1% additional parameters
- Faster training
- Enables multiple task-specific adapters on one base model
- Ideal for rapid iteration under compute constraints

---

## FEMA Guidelines → RAG Knowledge Base

### Source
~11 official FEMA manuals (planning, response, logistics, communications)

### Processing Pipeline
1. **PDF → Clean Text**
   - Remove headers, footers, boilerplate
   - Normalize whitespace

2. **Section-aware Chunking**
   - Preserve document titles, section IDs, page numbers

3. **Embedding & Indexing**
   - Sentence-Transformer embeddings
   - FAISS vector store

### Retrieval-Augmented Generation
- Triggered only when tweet signals indicate serious damage or requests
- Retrieved FEMA passages are injected into LLM prompts
- Output actions are traceable to specific guideline sections

---

## Agent Orchestration (LangGraph)

Built using **LangGraph / Pregel**:
- Conditional routing between agents
- Deterministic JSON outputs for safety-critical planning
- Offline-capable LLM support (Meta-Llama-3-8B via Ollama)

---

## Output Example (Operator Briefing)

```json
{
  "radar": {
    "tornado_probability": 0.73,
    "location": "Moore, OK"
  },
  "tweet_signal": {
    "damage": 0.61,
    "request": 0.22
  },
  "recommended_actions": [
    "Activate NIMS protocols per FEMA PA-428",
    "Notify State Emergency Operations Center within 2 hours",
    "Deploy mobile communications units to impacted sectors"
  ]
}
```

---

## Evaluation Highlights

- **Radar model:** ROC-AUC ≈ 0.86 (rare-event classification)
- **Tweet classifier:** >99% accuracy, Macro-F1 ≈ 0.99
- **End-to-end system:**
  - High recall prioritized (life-critical)
  - Precision controlled via threshold tuning
  - Time-aware validation to avoid data leakage

> **Note:** Accuracy alone is misleading in disaster detection — recall, PR-AUC, and operational behavior matter far more.

---

## Future Work

- [ ] Real-time ingestion (Kafka / Kinesis)
- [ ] Geo-spatial dashboards for first responders
- [ ] Inventory & supply chain coordination
- [ ] Extension to floods, wildfires, hurricanes
- [ ] Deployment via Docker / Kubernetes
- [ ] Human-in-the-loop feedback and continual learning

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python, TensorFlow, PyTorch |
| **NLP** | HuggingFace Transformers, PEFT (LoRA) |
| **Agents** | LangChain, LangGraph |
| **Vector DB** | FAISS |
| **Data** | Snorkel, XArray, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |

---

## Authors

**Muhammad Faizan Raza**
MS Data Analytics, Penn State
AI / ML / NLP Researcher

**Shuo (Luna) Yang**

**Matthew Varghese**

---

**⭐ If you find this project useful, please consider giving it a star!**
