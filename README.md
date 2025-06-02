# GraphRAG-V — Fast Multi-Hop Retrieval via Text-Chunk Communities
*A graph-free alternative to Microsoft’s GraphRAG that indexes in **minutes**, runs on a single GPU, and boosts multi-hop QA accuracy.*

---

## 1  Why GraphRAG-V?
*(unchanged — still includes the 37.9 → 48.8 % recall jump and 5-min index build)*

---

## 2  Repository contents

```

notebooks/
├─ MultiHopRAG.ipynb            # vector-store baseline ➊ + GraphRAG-V ➋
├─ MultiHop\_GraphRAG.ipynb      # Microsoft GraphRAG reference impl. (slow)
└─ QA\_Evaluation.ipynb          # merges run files & scores EM / Acc / F1
README.md

````

> **No `src/` folder yet** – all logic lives inside the notebooks.  
> ➊ *Vector RAG* uses FAISS + BGE-M3 embeddings.  
> ➋ *GraphRAG-V* re-uses those embeddings and adds VLouvain communities.

---

## 3  Prerequisites

| Item | Notes |
|------|-------|
| Python ≥ 3.10 | notebooks rely on recent LangChain + vLLM |
| **1 × A100 (40 GB / 80 GB)** | Colab Pro/Pro+ works; RTX 4090 also ok |
| **vLLM** | local OpenAI-compatible server (`pip install "vllm[nvidia]"`) |
| **Qwen-2.5-7B-Instruct** | default LLM in all notebooks (see launch cmd below) :contentReference[oaicite:1]{index=1} |

---

## 4  Start the local LLM once (own shell / Colab tab)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2_5_7B_Instruct \
  --port 8000 \
  --max-model-len 8192
````

*The notebooks talk to `http://localhost:8000/v1` and use `OPENAI_API_KEY="EMPTY"`.*&#x20;

---

## 5  Notebook walk-through

| Notebook                     | What happens                                                                                                                                                                                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MultiHopRAG.ipynb**        | 1. *Vector RAG baseline* on the full 2 556-question MultiHopRAG **Dev** set.<br>2. *GraphRAG-V* pipeline on the **same corpus**, answering the **same 2 556 Qs** unless you down-sample in `QUESTION_IDS`.<br>All index & query code is executed automatically. |
| **MultiHop\_GraphRAG.ipynb** | Microsoft’s original GraphRAG reference implementation. The entity-extract → graph-build → query loop is **very slow** (hours on a single GPU) but included for completeness and fair comparison.                                                               |
| **QA\_Evaluation.ipynb**     | Loads the JSONL answer files emitted by the first two notebooks, merges them if needed, and computes **Exact Match, Accuracy, Precision/Recall, and macro F1**.                                                                                                 |
## 6  Headline results (single-GPU Colab box)

| Metric         | Vector RAG | GraphRAG-V  | Δ                          |
| -------------- | ---------- | ----------- | -------------------------- |
| **Recall**     | 37.9 %     | **48.8 %**  | +10.9 pp                   |
| **Accuracy**   | 44.6 %     | **49.4 %**  | +4.8 pp                    |
| **Index time** | 74 s       | **5.3 min** | +4.6 min                   |
| **QA time**    | 12 min     | **42 min**  | +30 min (still ≪ GraphRAG) |

*Numbers from Tables 1–3, paper pp 8-9* .

---

## 7  How GraphRAG-V works (30-sec tour)

The *diagram on page 5* of the paper shows two retrieval paths:

1. **Local** – rank fine-level communities by cosine(query, summary) → retrieve passages inside top *k<sub>c</sub>* groups.
2. **Global** – rank *root* communities → have the LLM summarise each child branch → fuse three partial answers.

Both paths share **one query embedding**; extra overhead is three short LLM calls per query.

---

## 8  Customising

| Task                   | Where to edit                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Change model           | vLLM launch cmd + `ChatOpenAI` in each notebook                                      |
| More / fewer questions | `QUESTION_IDS` list in *MultiHop\_GraphRAG.ipynb*                                    |
| Memory tight?          | Reduce `--max-model-len`, choose a 7-B model, or set `--gpu-memory-utilization 0.85` |

---

## 9  Troubleshooting

| Problem                      | Fix                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| First LLM call hangs         | vLLM not running or wrong port                                                         |
| CUDA OOM during indexing     | lower batch size in GraphRAG CLI (`--batch-size 128`)                                  |
| Evaluation can’t find JSONLs | copy `answers_*.jsonl` + `sampled_indices.json` into same dir as `QA_Evaluation.ipynb` |

---

## 10  Citation

Please cite both the paper and this repo:

```bibtex
@inproceedings{yu2025graphragv,
  title     = {GraphRAG-V: Fast Multi-Hop Retrieval via Text-Chunk Communities},
  author    = {Tengkai Yu and Venkatesh Srinivasan and Alex Thomo},
  year      = {2025},
  url       = {https://github.com/yutengkai/GraphRAG-V}
}
