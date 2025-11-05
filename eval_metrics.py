import csv, time, os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore

from app_cli import build_context, ask_gpt, ask_gemini

load_dotenv()
DB_DIR = "chroma_db"
SYSTEM = open("prompts/qa_system.md").read()
vs = Chroma(persist_directory=DB_DIR)

def ask_with_backend(q, backend):
    docs = vs.similarity_search(q, k=4)
    ctx = build_context(docs)
    return ask_gpt(SYSTEM, q, ctx) if backend=="gpt" else ask_gemini(SYSTEM, q, ctx)

rows = list(csv.DictReader(open("eval/qa.csv")))
preds = {"gpt": [], "gemini": []}
refs  = [r["reference_answer"] for r in rows]

for r in rows:
    q = r["question"]
    for backend in ["gpt", "gemini"]:
        preds[backend].append(ask_with_backend(q, backend))
        time.sleep(0.2)

def rouge_avg(hyps):
    R = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    sums = {"rouge1":0,"rouge2":0,"rougeL":0}
    for h, ref in zip(hyps, refs):
        s = R.score(ref, h)
        for k in sums: sums[k]+=s[k].fmeasure
    n=len(hyps); return {k: v/n for k,v in sums.items()}

def bleu(hyps):
    return {
        "bleu1": sacrebleu.corpus_bleu(hyps, [refs], smooth_method="exp", max_ngram_order=1).score,
        "bleu4": sacrebleu.corpus_bleu(hyps, [refs]).score
    }

def bert(hyps):
    _,_,F = bertscore(hyps, refs, lang="en")
    return float(F.mean())

for b in ["gpt","gemini"]:
    print(f"\n=== {b.upper()} ===")
    print({**rouge_avg(preds[b]), **bleu(preds[b]), "bertscore_f1": bert(preds[b])})
