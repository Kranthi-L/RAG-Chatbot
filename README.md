# 📘 CS-297 Phase-1: Comparative Study and Implementation of RAG Chatbots

## 🎯 Goal

The goal of this project is to **reproduce and extend** the ideas from  
**“A Comparative Study of Retrieval-Augmented Generation (RAG) Chatbots”**  
by evaluating how different large language models (LLMs) perform when grounded on the same domain corpus using a consistent Retrieval-Augmented Generation (RAG) pipeline.

This project compares **GPT-4o-mini (OpenAI)** and **Claude-3.5 (Anthropic)** on academic textbooks and measures how accurately they answer questions using only retrieved context.

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

---

## 🧠 Overview of Project Phases

| Phase       | Description                                                                                                           | Status       |
| ----------- | --------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Phase 1** | Build a RAG chatbot using course textbooks, compare GPT and Claude using evaluation metrics (ROUGE, BLEU, BERTScore). | ✅ Completed |
| **Phase 2** | Add accessibility features like speech input/output and visual customization for differently-abled learners.          | 🚧 Planned   |
| **Phase 3** | Implement adaptive learning capabilities for personalized educational experiences.                                    | 🚧 Planned   |

This phase (Phase 1) reproduces the paper’s experimental setup and extends it with persistent sessions, follow-up question reasoning, and a web-based interface.

---

## 📄 Reference Paper

**Title:** _A Comparative Study of Retrieval-Augmented Generation (RAG) Chatbots_  
**Objective:** Compare multiple LLMs (GPT, Gemini, etc.) under the same RAG pipeline using ROUGE, BLEU, and BERTScore metrics.

**Key Findings from the Paper:**

- Retrieval grounding improved factual accuracy by 15–25%.
- GPT-style models showed higher lexical overlap; other models were semantically comparable.
- BERTScore best reflected semantic correctness across all chatbots.

Your implementation follows the same methodology and achieves results within the same metric range.

---

## 🚀 Current Progress and Achievements

✅ Indexed two full textbooks:

- _Computer Networking: A Top-Down Approach_
- _Computer Architecture: A Quantitative Approach_

✅ Built a complete RAG pipeline (Ingestion → Embedding → Retrieval → Generation).  
✅ Compared GPT vs Claude quantitatively using ROUGE, BLEU, and BERTScore.  
✅ Implemented persistent conversation sessions and follow-up question understanding.  
✅ Created both a Command-Line Interface (CLI) and a Streamlit web app.  
✅ Achieved metric results consistent with those reported in the paper.

**Evaluation Summary (Networking Dataset):**

| Model       | ROUGE-L | BLEU-4 | BERTScore (F1) | Interpretation                                  |
| ----------- | ------- | ------ | -------------- | ----------------------------------------------- |
| GPT-4o-mini | 0.169   | 2.80   | 0.843          | Strong semantic match; close to ideal phrasing. |
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

📊 Evaluation Metrics Explained
Metric Purpose Interpretation
ROUGE-1/2/L Word and phrase overlap between model and ideal answers. High values = similar phrasing.
BLEU-1/4 N-gram precision, measures exact phrase matches. High values = closer wording.
BERTScore (F1) Semantic similarity using contextual embeddings. High values = meaning preserved even if rephrased.

In explanatory Q&A, high BERTScore (≥0.8) but moderate ROUGE/BLEU indicates correct meaning with varied wording — the expected pattern in RAG chatbot evaluations.

📈 Results Summary

Both GPT and Claude achieved semantic similarity (BERTScore ≈ 0.83–0.84), aligning with the paper’s reported results.

GPT scored slightly higher on ROUGE and BLEU, showing tighter phrasing adherence.

Follow-up question support enables pronoun resolution (“that,” “it,” “this”) and true multi-turn reasoning.

Metrics validate that your chatbot performs comparably to the paper’s systems.

🧠 Features Implemented
Feature Description Status
PDF ingestion & vector DB Converts textbooks to searchable embeddings. ✅
RAG pipeline Retrieve relevant context and generate grounded answers. ✅
Multi-model comparison GPT vs Claude under identical setup. ✅
Evaluation metrics Automatic scoring (ROUGE, BLEU, BERTScore). ✅
Session persistence Save and resume chats by session ID. ✅
Follow-up understanding Summarize and rewrite follow-up questions. ✅
Streamlit web UI Interactive, user-friendly interface. ✅
Accessibility (Speech I/O, font control) Voice input/output for differently-abled users. 🚧 Phase 2
Adaptive learning Personalized tutoring and progress tracking. 🚧 Phase 3
🔭 Next Phases
Phase 2 — Accessibility & Multimodality

Goal: Make the chatbot inclusive and easy to use for differently-abled users.

🗣️ Speech-to-Text (STT): Convert spoken questions to text (Whisper).

🔊 Text-to-Speech (TTS): Read out chatbot responses (pyttsx3 or macOS say).

🎨 Visual Accessibility: Font resizing, color contrast modes, and high-contrast UI.

♿ Keyboard-only navigation and ARIA labels for accessibility compliance.

Phase 3 — Adaptive Learning & Analytics

Goal: Turn the chatbot into a personalized learning companion.

📈 Track user performance across sessions.

🎯 Identify weak topics and generate targeted follow-up questions.

🧩 Dynamically adjust explanation depth and difficulty.

📚 Integrate analytics dashboard for instructors.
