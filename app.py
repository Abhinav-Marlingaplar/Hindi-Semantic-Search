"""
Hindi question search: keyword vs semantic retrieval (interactive demo).

Standalone version of Section 14 of the project notebook
"Semantic Search for Hindi Documents using Sentence Embeddings".

It searches the 5,046 Hindi questions of Bharat_NanoQuoraRetrieval_hi with the six
methods compared in the project (TF-IDF default/fixed, BM25, MiniLM, multilingual
E5-base and a hybrid), and lets you explore the 50 evaluation queries.

Usage
    python app.py                 # local app at http://127.0.0.1:7860
    python app.py --share         # also create a temporary public link
    python app.py --port 8000

First run: downloads the dataset and the two encoders and encodes the corpus
(a few minutes on CPU, much faster on a GPU). Document embeddings are cached in ./cache
so later runs start in seconds.
"""
import argparse
import html
import inspect
import os
import re
import unicodedata

import numpy as np

# ----------------------------------------------------------------------------- configuration
REPO = "carlfeynman/Bharat_NanoQuoraRetrieval_hi"
CACHE_DIR = "cache"
TOKEN_PATTERN = r"[^\s\?\.,!।:;\"'()]+"          # keeps Devanagari vowel signs attached to their word
RRF_K = 60
MARGIN = 0.10                                    # a method "wins" a query if it leads by >= 0.10 nDCG@10

DENSE = {
    "minilm": dict(name="paraphrase-multilingual-MiniLM-L12-v2", qp="", dp=""),
    "e5": dict(name="intfloat/multilingual-e5-base", qp="query: ", dp="passage: "),
}
NAMES = {  # display name -> engine key
    "TF-IDF (midterm)": "tfidf_mid",
    "TF-IDF (fixed)": "tfidf",
    "BM25": "bm25",
    "MiniLM": "minilm",
    "mE5-base": "e5",
    "Hybrid RRF": "hybrid",
}
HYBRID_COMPONENTS = ["bm25", "e5"]               # BM25 + the stronger encoder (as selected in the notebook)
DEFAULT_METHODS = ["TF-IDF (fixed)", "BM25", "mE5-base", "Hybrid RRF"]
CATS = ["Semantic wins", "Lexical wins", "Comparable – both solve", "Comparable – hard"]


# ----------------------------------------------------------------------------- text helpers
def clean_text(t):
    """Same preprocessing as the notebook: Unicode NFC + whitespace cleanup."""
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def tok(s):
    return re.findall(TOKEN_PATTERN, s.lower())


def _rank_vec(s):
    """Rank of every document (1 = best) for a score vector."""
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r


def ndcg_at_k(ranked, rel, k=10):
    dcg = lambda gains: sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------------- retrieval engine
class Engine:
    """Loads the data, builds all indexes and pre-computes the evaluation rankings."""

    def __init__(self):
        from datasets import load_dataset
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("[1/4] Loading dataset ...")
        corpus_df = load_dataset(REPO, "corpus")["train"].to_pandas()
        queries_df = load_dataset(REPO, "queries")["train"].to_pandas()
        qrels_df = load_dataset(REPO, "qrels")["train"].to_pandas()
        for df in (corpus_df, queries_df):
            df["_id"] = df["_id"].astype(str)
            df["text_clean"] = df["text"].apply(clean_text)

        def find(cols, *keys):
            for c in cols:
                if any(k in c.lower() for k in keys):
                    return c
            raise KeyError(f"No column matching {keys} in {list(cols)}")

        qcol = find(qrels_df.columns, "query")
        dcol = find([c for c in qrels_df.columns if c != qcol], "corpus", "doc")
        scol = next((c for c in qrels_df.columns if c not in (qcol, dcol)
                     and any(k in c.lower() for k in ("score", "rel", "label"))), None)
        self.qrels = {}
        for _, r in qrels_df.iterrows():
            gain = int(r[scol]) if scol else 1
            if gain > 0:
                self.qrels.setdefault(str(r[qcol]), {})[str(r[dcol])] = gain

        self.doc_ids = corpus_df["_id"].tolist()
        self.texts = corpus_df["text_clean"].tolist()
        self.text_by_id = dict(zip(self.doc_ids, self.texts))
        self.n_docs = len(self.texts)
        self.qids = [q for q in queries_df["_id"] if q in self.qrels]
        self.qtexts = queries_df.set_index("_id").loc[self.qids, "text_clean"].tolist()
        self.qt2qid = {t: q for t, q in zip(self.qtexts, self.qids)}

        text2idx = {}
        for i, t in enumerate(self.texts):
            text2idx.setdefault(t, []).append(i)
        # identical copy of the query that is not a labelled answer = self-match (excluded, as in the evaluation)
        self.self_excl = {q: [i for i in text2idx.get(t, []) if self.doc_ids[i] not in self.qrels[q]]
                          for q, t in zip(self.qids, self.qtexts)}

        print("[2/4] Building keyword indexes ...")
        self.vec = {"tfidf_mid": TfidfVectorizer(), "tfidf": TfidfVectorizer(token_pattern=TOKEN_PATTERN)}
        self.mat = {k: v.fit_transform(self.texts) for k, v in self.vec.items()}
        self.bm25 = BM25Okapi([tok(t) for t in self.texts])

        print("[3/4] Loading encoders and document embeddings (first run encodes the corpus) ...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.models, self.emb = {}, {}
        for key, cfg in DENSE.items():
            self.models[key] = SentenceTransformer(cfg["name"])
            path = os.path.join(CACHE_DIR, f"{key}_docs.npy")
            cached = np.load(path) if os.path.exists(path) else None
            if cached is None or cached.shape[0] != self.n_docs:
                cached = self.models[key].encode([cfg["dp"] + t for t in self.texts], batch_size=64,
                                                 normalize_embeddings=True, show_progress_bar=True)
                np.save(path, cached)
            self.emb[key] = cached

        print("[4/4] Pre-computing evaluation rankings for the 50 queries ...")
        self._evaluate()
        print("Ready.")

    # --- scoring (identical logic to the notebook) ---
    def base_scores(self, query, method):
        q = clean_text(query)
        if method in self.vec:
            return (self.vec[method].transform([q]) @ self.mat[method].T).toarray().ravel()
        if method == "bm25":
            return np.asarray(self.bm25.get_scores(tok(q)), dtype=float)
        if method in DENSE:
            e = self.models[method].encode([DENSE[method]["qp"] + q], normalize_embeddings=True,
                                           show_progress_bar=False)[0]
            return self.emb[method] @ e
        raise ValueError(f"unknown method {method}")

    def score_vector(self, query, method, exclude=()):
        exclude = list(exclude)
        if method == "hybrid":
            s = np.zeros(self.n_docs)
            for comp in HYBRID_COMPONENTS:
                b = self.base_scores(query, comp).astype(float)
                if exclude:
                    b[exclude] = -np.inf
                s += 1.0 / (RRF_K + _rank_vec(b))
        else:
            s = self.base_scores(query, method).astype(float)
        if exclude:
            s[exclude] = -np.inf
        return s

    def search(self, query, method, k=5, exclude=()):
        s = self.score_vector(query, method, exclude)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.doc_ids[i], self.texts[i], float(s[i])) for i in top]

    # --- evaluation rankings used by the Query explorer ---
    def _evaluate(self, topk=100):
        self.rankings = {n: {} for n in NAMES}
        self.ndcg = {n: {} for n in NAMES}
        for name, key in NAMES.items():
            for q, qt in zip(self.qids, self.qtexts):
                s = self.score_vector(qt, key, exclude=self.self_excl[q])
                ranked = [self.doc_ids[i] for i in np.argsort(-s, kind="stable")[:topk]]
                self.rankings[name][q] = ranked
                self.ndcg[name][q] = ndcg_at_k(ranked, self.qrels[q], 10)
        self.category = {}
        for q in self.qids:
            lex, sem = self.ndcg["BM25"][q], self.ndcg["mE5-base"][q]
            d = sem - lex
            if d >= MARGIN:
                self.category[q] = CATS[0]
            elif d <= -MARGIN:
                self.category[q] = CATS[1]
            elif lex >= 0.9 and sem >= 0.9:
                self.category[q] = CATS[2]
            else:
                self.category[q] = CATS[3]


# ----------------------------------------------------------------------------- HTML rendering
STYLE = """<style>
.hs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin-top:8px}
.hs-col{border:1px solid rgba(128,128,128,.45);border-radius:4px;padding:8px 12px;min-width:0}
.hs-col h4{margin:0 0 6px 0;padding-bottom:6px;border-bottom:1px solid rgba(128,128,128,.45);font-size:.95rem}
.hs-row{display:flex;gap:8px;padding:5px 0;border-bottom:1px dotted rgba(128,128,128,.35);font-size:.95rem;line-height:1.4}
.hs-row:last-child{border-bottom:none}
.hs-rank{opacity:.6;min-width:1.3em}
.hs-text{flex:1}
.hs-score{opacity:.55;font-size:.8rem;white-space:nowrap}
.hs-hit .hs-text{font-weight:600}
.hs-note{opacity:.8;font-size:.9rem;margin:6px 0}
.hs-table{border-collapse:collapse;width:100%;font-size:.95rem;margin:8px 0;border:none!important}
.hs-table tr,.hs-table thead,.hs-table tbody{border:none!important;box-shadow:none!important}
.hs-table th,.hs-table td{border:none!important;border-bottom:1px solid rgba(128,128,128,.4)!important;padding:5px 8px;text-align:left}
</style>"""


def _column(title, rows, relevant=None):
    """One method's result list. rows = [(doc_id, text, score_or_None)]."""
    parts = ['<div class="hs-col"><h4>' + html.escape(title) + "</h4>"]
    for rank, (did, text, score) in enumerate(rows, 1):
        hit = relevant is not None and did in relevant
        cls = "hs-row hs-hit" if hit else "hs-row"
        mark = "✔ " if hit else ""
        sc = "" if score is None else '<span class="hs-score">' + f"{score:.3f}" + "</span>"
        parts.append('<div class="' + cls + '"><span class="hs-rank">' + str(rank) + "</span>"
                     '<span class="hs-text">' + mark + html.escape(text) + "</span>" + sc + "</div>")
    parts.append("</div>")
    return "".join(parts)


# ----------------------------------------------------------------------------- Gradio app
def build_app(eng):
    import gradio as gr

    def run_search(query, methods, k):
        q = clean_text(query)
        if not q:
            return STYLE + '<div class="hs-note">Type a Hindi question above.</div>'
        if not methods:
            return STYLE + '<div class="hs-note">Select at least one method.</div>'
        qid = eng.qt2qid.get(q)
        excl = eng.self_excl[qid] if qid else []
        relevant = set(eng.qrels[qid]) if qid else None
        cols = [_column(m, eng.search(q, NAMES[m], int(k), exclude=excl), relevant) for m in methods]
        if qid:
            note = (f"This is evaluation query {qid}. A ✔ marks a question labelled as relevant. "
                    "An identical copy of the query is left out, as in the evaluation.")
        else:
            note = ("Free-text query, so no labels are available and results are not marked. "
                    "Scores are not comparable across methods (cosine, TF-IDF, BM25 and RRF scores differ in scale).")
        return STYLE + '<div class="hs-note">' + html.escape(note) + '</div><div class="hs-grid">' + "".join(cols) + "</div>"

    cat_order = ["All"] + CATS

    def choices(cat):
        return [f"{q} · {t[:70]}" for q, t in zip(eng.qids, eng.qtexts) if cat == "All" or eng.category[q] == cat]

    def explore(choice):
        if not choice:
            return STYLE + '<div class="hs-note">Pick an evaluation query.</div>'
        qid = choice.split(" · ")[0]
        qt, rel = eng.qtexts[eng.qids.index(qid)], eng.qrels[qid]
        parts = [STYLE,
                 '<div class="hs-note"><b>Query ' + html.escape(qid) + "</b> · outcome: <b>"
                 + html.escape(eng.category[qid]) + "</b></div>",
                 '<div style="font-size:1.1rem;margin:4px 0 10px 0">' + html.escape(qt) + "</div>",
                 '<div class="hs-note"><b>Labelled relevant question(s)</b></div>']
        for d in rel:
            parts.append('<div class="hs-row"><span class="hs-text">' + html.escape(eng.text_by_id[d]) + "</span></div>")
        trs = []
        for name in NAMES:
            first = next((i + 1 for i, d in enumerate(eng.rankings[name][qid]) if d in rel), None)
            trs.append("<tr><td>" + html.escape(name) + "</td><td>" + (str(first) if first else "&gt;100")
                       + "</td><td>" + f"{eng.ndcg[name][qid]:.2f}" + "</td></tr>")
        parts.append('<table class="hs-table"><tr><th>Method</th><th>Rank of first relevant</th><th>nDCG@10</th></tr>'
                     + "".join(trs) + "</table>")
        parts.append('<div class="hs-note"><b>Top 3 results per method</b> (✔ = labelled relevant)</div><div class="hs-grid">')
        for name in NAMES:
            rows = [(d, eng.text_by_id[d], None) for d in eng.rankings[name][qid][:3]]
            parts.append(_column(name, rows, set(rel)))
        parts.append("</div>")
        return "".join(parts)

    def by_qid(qid):
        return eng.qtexts[eng.qids.index(qid)] if qid in eng.qids else None

    examples = [t for t in (by_qid("362950"), by_qid("155873"), by_qid("315712"),
                            next((t for t in eng.qtexts if "स्क्रम मास्टर" in t), None),
                            next((t for t in eng.qtexts if "इतनी तेजी से क्यों गुजरता" in t), None)) if t]
    examples.append("दिन इतनी जल्दी क्यों बीत जाते हैं?")        # free text, not in the dataset

    theme = gr.themes.Base(primary_hue="slate", neutral_hue="slate")
    theme_in_launch = "theme" in inspect.signature(gr.Blocks.launch).parameters   # Gradio 6 moved theme to launch()

    with gr.Blocks(title="Hindi question search", **({} if theme_in_launch else {"theme": theme})) as demo:
        gr.Markdown("## Hindi question search: keyword vs semantic retrieval\n"
                    "Searches the 5,046 Hindi questions of the project dataset with the methods compared in the project.")
        with gr.Tabs():
            with gr.Tab("Search"):
                query = gr.Textbox(label="Hindi question", placeholder="उदाहरण: दिन इतनी जल्दी क्यों बीत जाते हैं?", lines=1)
                with gr.Row():
                    methods = gr.CheckboxGroup(list(NAMES), value=DEFAULT_METHODS, label="Methods")
                    topk = gr.Slider(3, 10, value=5, step=1, label="Results per method")
                go = gr.Button("Search", variant="primary")
                out = gr.HTML()
                gr.Examples(examples=[[t] for t in examples], inputs=[query],
                            label="Examples (click one, then press Search)")
                go.click(run_search, [query, methods, topk], out)
                query.submit(run_search, [query, methods, topk], out)
            with gr.Tab("Query explorer"):
                cat = gr.Radio(cat_order, value="All",
                               label="Show evaluation queries by outcome (BM25 vs the best embedding model)")
                pick = gr.Dropdown(choices("All"), label="Evaluation query")
                view = gr.HTML()
                cat.change(lambda c: gr.update(choices=choices(c), value=None), cat, pick)
                pick.change(explore, pick, view)

    demo.run_search, demo.explore, demo.choices = run_search, explore, choices      # exposed for testing
    demo.launch_kwargs = {"theme": theme} if theme_in_launch else {}
    return demo


def main():
    ap = argparse.ArgumentParser(description="Hindi question search demo (keyword vs semantic retrieval).")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860, help="port (default 7860)")
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = ap.parse_args()
    demo = build_app(Engine())
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, **demo.launch_kwargs)


if __name__ == "__main__":
    main()
