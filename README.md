# Semantic Search for Hindi Documents using Sentence Embeddings
### A Comparative Study of TF-IDF and Semantic Retrieval

**CSE413 Speech and Natural Language Processing**

**Team:** 

Abhinav Marlingaplar (2023BCD0013)

Saksham Saklani (2023BCD0049)

## What this project does
A search system should find documents that match what a user *means*, not only the words they typed. We compare two ways of searching Hindi text:

- **Keyword retrieval:** TF-IDF and BM25 score documents by the words they share with the query.
- **Semantic retrieval:** sentence embeddings (MiniLM and multilingual E5) turn each sentence into a vector that captures its meaning, so a paraphrase can match even when no word is shared.

We evaluate six methods (TF-IDF default, TF-IDF fixed, BM25, MiniLM, E5-base and a BM25+E5 hybrid) on a Hindi duplicate-question collection with 5,046 questions, 50 queries and 70 relevance judgments. Quality is measured with nDCG@10, MRR@10 and Recall@10, with bootstrap confidence intervals and paired significance tests.

## Key results
| Method | nDCG@10 [95% CI] | MRR@10 | Recall@10 |
|---|---|---|---|
| TF-IDF (default) | 0.649 [0.547, 0.751] | 0.619 | 0.816 |
| TF-IDF (fixed tokenizer) | 0.692 [0.585, 0.790] | 0.676 | 0.816 |
| BM25 | 0.658 [0.548, 0.763] | 0.649 | 0.773 |
| MiniLM | 0.845 [0.765, 0.913] | 0.826 | 0.953 |
| **Multilingual E5-base** | **0.893** [0.826, 0.949] | **0.877** | **0.977** |
| Hybrid (RRF of BM25 and E5) | 0.775 [0.678, 0.859] | 0.747 | 0.903 |

- Embedding models beat BM25 by margins that survive a multiple-comparison correction (E5: +0.235, MiniLM: +0.187 nDCG@10).
- Equal-weight fusion of BM25 and E5 is significantly *worse* than E5 alone (-0.118).
- scikit-learn's default tokenizer cuts Devanagari words into fragments; fixing it gave no statistically clear gain on 50 queries.
- E5 vs MiniLM, and BM25 vs TF-IDF, are inconclusive with 50 queries.

Full details are in the report.

## Repository contents
```
hindi-semantic-search/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                          # interactive demo (Search + Query explorer)
├── Hindi_Semantic_Search_Final.ipynb   # full implementation, experiments and analysis
├── Hindi_Semantic_Search_Final_Report.pdf 
├── Hindi_Semantic_Search_Final_Presentation.pptx
└──demo
    ├── demo_search/                    # screenshots of the Search tab
    │   ├── search1.png
    │   └── search2.png
    └──demo_explorer/                  # screenshots of the Query explorer tab
        ├── explorer1.png
        └── explorer2.png
```

## Setup

### Option 1: Google Colab (recommended for the notebook)
1. Upload `notebook/Hindi_Semantic_Search_Final.ipynb` to Colab.
2. Choose **Runtime > Change runtime type > GPU**.
3. Run all cells. The notebook installs its own dependencies, downloads the dataset and models, and writes its outputs to `results/`. The last section launches the interactive demo and prints a link.

### Option 2: Run locally
Requires Python 3.9 or newer.
```bash
git clone https://github.com/<your-username>/hindi-semantic-search.git
cd hindi-semantic-search
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the interactive demo
```bash
python app.py                # opens at http://127.0.0.1:7860
python app.py --share        # also creates a temporary public link
python app.py --port 8000
```
- **First run:** downloads the dataset and the two encoders (about 1.2 GB) and encodes the 5,046 questions. This takes a few minutes on CPU and about a minute on a GPU. Embeddings are cached in `cache/`, so later runs start in seconds.
- **Search tab:** type a Hindi question and compare the top results of the selected methods side by side. For the 50 evaluation queries, a check mark shows the questions labelled as relevant.
- **Query explorer tab:** pick any evaluation query to see its relevant question(s), the rank each method gives them, and each method's top 3 results.

The demo searches only the 5,046 questions of the project dataset.

## Demo screenshots

**Search tab:** type a Hindi question and compare the results of the selected methods side by side.

<table>
  <tr>
    <td width="50%"><img src="demo/demo_search/demo_search1.png" alt="Search tab, screenshot 1" width="100%"></td>
    <td width="50%"><img src="demo/demo_search/demo_search2.png" alt="Search tab, screenshot 2" width="100%"></td>
  </tr>
</table>

**Query explorer tab:** pick an evaluation query to see its relevant question(s), each method's rank for them, and each method's top 3 results.

<table>
  <tr>
    <td width="50%"><img src="demo/demo_explorer/demo_explorer1.png" alt="Query explorer tab, screenshot 1" width="100%"></td>
    <td width="50%"><img src="demo/demo_explorer/demo_explorer2.png" alt="Query explorer tab, screenshot 2" width="100%"></td>
  </tr>
</table>

## Reproducing the results
All numbers in the report come from the notebook (fixed random seed 42 for the bootstrap). Running it top to bottom regenerates the tables in `results/`: `summary.csv`, `paired_tests.csv`, `per_query_ndcg10.csv`, `per_query_category.csv` and `config.json`.

## Limitations
Only 50 queries on one dataset of short, machine-translated Hindi questions; labels are incomplete (about 1.4 relevant questions per query); no model was fine-tuned. See the report for details.

## Data and models
- Dataset: [`carlfeynman/Bharat_NanoQuoraRetrieval_hi`](https://huggingface.co/datasets/carlfeynman/Bharat_NanoQuoraRetrieval_hi)
- Models: `paraphrase-multilingual-MiniLM-L12-v2`, `intfloat/multilingual-e5-base`
- Motivated by Hindi-BEIR / NLLB-E5 (Acharya et al., NAACL 2025) and IndicIRSuite (Haq et al., ACL 2024).
