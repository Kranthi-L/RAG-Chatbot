# 📘 CS-297 Phase-1: Comparative Study and Implementation of RAG Chatbots

## 🎯 Goal

The goal of this project is to **reproduce and extend** the ideas from  
**“A Comparative Study of Retrieval-Augmented Generation (RAG) Chatbots”**  
by evaluating how different large language models (LLMs) perform when grounded on the same domain corpus using a consistent Retrieval-Augmented Generation (RAG) pipeline.

This project compares **GPT-5 (OpenAI)** and **Claude-3.5 (Anthropic)** on academic textbooks and measures how accurately they answer questions using only retrieved context.

---

## 💡 Motivation

General-purpose chatbots like ChatGPT or Claude can answer broad questions but often lack **grounding** to specific sources, which limits factual accuracy in specialized domains like Computer Networking or Computer Architecture.

Retrieval-Augmented Generation (RAG) solves this problem by combining:

- **Retrieval:** fetch relevant information chunks from a trusted corpus.
- **Generation:** let an LLM answer using those retrieved chunks as factual context.

This project explores:

1. How different LLMs perform on **identical retrieval pipelines**.
2. Whether the chatbot can handle **follow-up questions** using conversational context.
3. How the results compare to the findings in the reference paper.



## 📄 Reference Paper

**Title:** _A Comparative Study of Retrieval-Augmented Generation (RAG) Chatbots_  
**Objective:** Compare multiple LLMs (GPT, Gemini, etc.) under the same RAG pipeline using ROUGE, BLEU, and BERTScore metrics.

**Key Findings from the Paper:**

- Retrieval grounding improved factual accuracy by 15–25%.
- GPT-style models showed higher lexical overlap; other models were semantically comparable.
- BERTScore best reflected semantic correctness across all chatbots.

Your implementation follows the same methodology and achieves results within the same metric range.

---

**Evaluation Summary (Networking Dataset):**

| Model       | ROUGE-L | BLEU-4 | BERTScore (F1) | Interpretation                                  |
| ----------- | ------- | ------ | -------------- | ----------------------------------------------- |
| GPT-5       | 0.169   | 2.80   | 0.843          | Strong semantic match; close to ideal phrasing. |
| Claude-3.5  | 0.145   | 2.51   | 0.833          | Semantically accurate; phrasing more narrative. |

---

## 📂 Project Structure

rag-chatbot/
├── app_cli.py # CLI chatbot (sessions + follow-ups)
├── app_streamlit.py # Simple single-turn Streamlit UI (baseline)
├── app_web.py # Enhanced Streamlit UI (sessions + follow-ups)
├── ingest.py # PDF ingestion, chunking, embedding, and indexing
├── memory.py # Session persistence (save/load conversations)
├── prompts/
│ └── qa_system.md # System prompt restricting answers to provided context
├── chroma_db/ # Vector database storing embedded document chunks
├── data/
│ ├── networking/ # Networking course PDFs
│ └── architecture/ # Architecture course PDFs
├── eval/
│ ├── networking_eval.csv # Question set + ideal answers
│ ├── networking_eval_filled.csv # Model answers
│ ├── networking_metrics.csv # Evaluation results (ROUGE/BLEU/BERTScore)
│ ├── run_batch_networking.py # Script to batch-generate GPT/Claude answers
│ └── eval_metrics.py # Script to calculate evaluation metrics
├── sessions/ # JSON session logs (persistent conversation memory)
├── requirements.txt # Project dependencies
└── .env # API keys for GPT (OpenAI) and Claude (Anthropic)

---

## ⚙️ Setup and Installation

# Clone repository

git clone <repo_url>
cd rag-chatbot

# Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate # (use .venv\Scripts\activate on Windows)

# Install dependencies

pip install -r requirements.txt

# Install FFmpeg (required for speech-to-text)
# macOS:
brew install ffmpeg

# Linux (Ubuntu/Debian):
# sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html and add to PATH

# How to Use the Project

🧱 Step 1 — Ingest PDFs
python ingest.py

This script:

Extracts text from PDFs under /data/.

Splits text into chunks.

Creates vector embeddings.

Saves them in a local database (/chroma_db/).

💬 Step 2 — Chatbot (Command-Line Interface)

Start or resume a session:

python app_cli.py gpt networking --session study1

Examples:

What is TCP congestion control?
How does that help fairness?
What are the drawbacks of congestion control?

Quit with q.
Conversations auto-save in /sessions/<session_id>.json.

🌐 Step 3 — Web Interface (Streamlit)
Baseline interface (single-turn chatbot):
streamlit run app_streamlit.py

Enhanced interface (sessions + follow-ups):
streamlit run app_web.py

Features:

Choose backend: GPT, Claude, or Both.

Filter by course: networking, architecture, or all.

Adjust retrieval depth (Top-K) and temperature.

Retrieve and display source filenames and page numbers.

Save and reload conversations by Session ID.

🧮 Step 4 — Evaluation Pipeline

1️⃣ Generate model responses:

python eval/run_batch_networking.py

2️⃣ Compute evaluation metrics:

python eval/eval_metrics.py

Outputs:

networking_eval_filled.csv — model-generated answers.

networking_metrics.csv — computed ROUGE, BLEU, and BERTScore metrics.

Console summary showing average scores per model.


---

# 🧭 Quick Reference (What Each File Does)

**Core RAG**
- `rag_core.py`: Vector store/BM25 loaders; retrievers (dense, bm25, hybrid, section_aware); learner-level personalization (beginner=12, intermediate=base, advanced=4) with rerank; generation style instructions. Functions: `retrieve_docs`, `build_context`, `answer_with_model`, `ask_gpt`, `ask_claude`.
- `ingest.py`: Load PDFs/PPTX from `data/<course>/`, chunk, tag metadata (`course`, `filename`, `page`, `level`, `concept_type`), build Chroma and per-course BM25 (`chroma_db/bm25_*.pkl`).
- `course_utils.py`: Discover courses from `data/`.

**Apps**
- `app_web.py`: Streamlit UI; pick backend, course, retriever type, learner level, top_k, temperature; uses `rag_core`.
- `app_cli.py`: CLI chat via shared RAG pipeline.

**Batch Q&A + Judging**
- `eval/run_batch_eval.py`: Flags `--course --retriever_type --top_k --temperature --learner_level --max_workers --debug_first_n`. Outputs `filled_<course>_temp..._topk..._<retriever><level>.csv` with responses, timings, num_chunks, context_chars, retriever_type, learner_level. Prereq: run `ingest.py`.
- `eval/run_llm_judge.py`: LLM correctness judge → `judged_*.csv`.
- `eval/eval_metrics.py`: Aggregates judged files (dense/hybrid/bm25/section_aware) → `summary_metrics.csv`; infers retriever_type from filename.

**Retrieval Evaluation**
- `eval/build_relevance_labels.py`: LLM relevance scores → `eval/data/relevance_labels.jsonl` (labels 0/1/2, hybrid default).
- `eval/run_retrieval_eval.py`: Recall@k, MRR, nDCG per retriever/k → `eval/retrieval_metrics.csv`.

**Accessibility Pipeline**
- `eval/accessibility/build_accessibility_inputs.py`: Prepares judge inputs.
- `eval/accessibility/run_accessibility_judge.py`: Accessibility judge, preserves `retriever_type` → `accessibility_results.csv`.
- `eval/accessibility/select_best_configs.py`: Picks best (course, model) configs, currently filtered to `retriever_type=="hybrid"` → `best_configs_for_accessibility.csv`.
- `eval/accessibility/analyze_accessibility.py`: Summaries/plots.

**Ablation (Prompt Accessibility Study)**
- `eval/ablation/ablation_generate_answers.py`: Hybrid retrieval (top_k=8) baseline vs accessible answers → `ablation_answers.csv`.
- `eval/ablation/ablation_run_accessibility_judge.py`: Judges accessibility/correctness → `ablation_accessibility_results.csv`.
- `eval/ablation/ablation_analyze.py`: Summaries/plots → CSVs + `figures_ablation/`.

**Simulation / Adaptive (Prototype)**
- `adaptive/learner_model.py`, `adaptive/adaptive_answer.py`, `adaptive/simulate_learners.py`, `adaptive/inspect_simulations.py`: Exploratory; not used in main results.

**Plotting**
- `plots/generate_plots.py`: Builds figures from `summary_metrics.csv` into `figures/`.

**Scripts for Phase 3**
- `run_phase3_hybrid_evals.sh`: Runs hybrid batch evals (top_k=8, temps {0.2,0.5,0.8}, all courses, max_workers=1).

---

# ▶️ How to Run (Typical Workflow)
1) **Ingest corpus**: `python ingest.py` (needs PDFs/PPTX in `data/<course>/`; outputs `chroma_db/`, BM25 pkls).
2) **Generate answers (hybrid example)**: `bash run_phase3_hybrid_evals.sh` → `filled_*_hybrid.csv`.
3) **Judge correctness**: `python -m eval.run_llm_judge` → `judged_*_hybrid.csv`.
4) **Aggregate metrics**: `python -m eval.eval_metrics` → `summary_metrics.csv`.
5) **Pick best accessibility configs (hybrid)**: `python -m eval.accessibility.select_best_configs` → `best_configs_for_accessibility.csv`.
6) **Accessibility judging**: `python -m eval.accessibility.run_accessibility_judge` → `accessibility_results.csv`.
7) **Plotting**: `python plots/generate_plots.py` → `figures/*.png`.
8) **Ablation (optional)**: generate/judge/analyze via scripts in `eval/ablation/`.
9) **Web app**: `streamlit run app_web.py` (after ingestion); choose course, retriever type, learner level, top_k, temperature.

---

# ✅ Outputs Cheat Sheet
- `chroma_db/`: Chroma + BM25 indexes.
- `filled_*.csv`: Model responses, timings, retriever_type, learner_level, num_chunks, context_chars.
- `judged_*.csv`: LLM judge scores.
- `summary_metrics.csv`: Aggregated metrics (bleu, rouge_l, bert_sim, avg_judge_score, retriever_type).
- `relevance_labels.jsonl`: Retrieval labels.
- `retrieval_metrics.csv`: Recall@k, MRR, nDCG per retriever/k.
- `best_configs_for_accessibility.csv`: Current best (hybrid) configs.
- `accessibility_results.csv`: Accessibility judge outputs.
- `figures/*.png`: Plots from summary_metrics.
- Ablation: `ablation_answers.csv`, `ablation_accessibility_results.csv`, `ablation_summary_by_model_condition.csv`, `ablation_deltas_*.csv`, `figures_ablation/*`.
