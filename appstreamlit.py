import os
import re
import sys
import pickle
import logging
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st

import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# LOGGING
def setup_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("wsd_app")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        fh = logging.FileHandler("logs/app.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


LOGGER = setup_logger()

# PAGE / STYLE
st.set_page_config(page_title="WSD Demo", page_icon="🧠", layout="wide")

CUSTOM_CSS = """
<style>
/* page padding */
.block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; }

/* subtle header */
.app-hero {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 18px;
  padding: 1.2rem 1.3rem;
  background: linear-gradient(135deg, rgba(240, 244, 255, 0.75), rgba(250, 250, 252, 0.9));
  margin-bottom: 1.1rem;
}
.app-hero h1 { margin: 0 0 .25rem 0; font-size: 1.65rem; }
.app-hero p  { margin: 0; color: rgba(49, 51, 63, 0.72); }

/* cards */
.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 16px;
  padding: 1.05rem 1.1rem;
  background: white;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);
}
.card-title {
  font-weight: 800;
  font-size: 1.05rem;
  margin-bottom: .55rem;
  display: flex;
  align-items: center;
  gap: .45rem;
}
.badge {
  font-size: 0.78rem;
  padding: 0.12rem 0.5rem;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.12);
  color: rgba(49, 51, 63, 0.7);
  background: rgba(249, 250, 251, 1);
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.92rem;
}
.small {
  font-size: 0.9rem;
  color: rgba(49, 51, 63, 0.72);
}
.kv {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: .25rem .75rem;
  margin-top: .5rem;
}
.k { color: rgba(49, 51, 63, 0.62); font-size: .9rem; }
.v { font-weight: 650; }
hr { margin: 1.1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="app-hero">
  <h1>🧠 Word Sense Disambiguation Demo</h1>
  <p>Compare <b>Lesk (extended)</b> vs <b>SVM</b> vs <b>BERT + cosine</b> on your own sentence.</p>
</div>
""",
    unsafe_allow_html=True,
)

# NLTK (one-time download on Streamlit Cloud)
@st.cache_resource
def ensure_nltk():
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("stopwords")
    return True


ensure_nltk()

# TOKENIZE / VALIDATE 
def simple_tokenize_like_dataset(sentence: str) -> List[str]:
    # keep punctuation as separate tokens (helps indexing)
    sentence = re.sub(r"([.,!?;:()\"'])", r" \1 ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence.split()


def validate_input(sentence: str, target_index: int) -> List[str]:
    toks = simple_tokenize_like_dataset(sentence)
    if not toks:
        raise ValueError("Sentence is empty.")
    if target_index < 0 or target_index >= len(toks):
        raise ValueError(
            f"target_index out of range. Got {target_index}, but sentence has {len(toks)} tokens."
        )
    return toks


def normalize_pos(wn_pos: Optional[str]) -> Optional[str]:
    if wn_pos is None:
        return None
    wn_pos = wn_pos.strip().lower()
    POS_MAP = {"noun": "n", "verb": "v", "adj": "a", "adv": "r", "n": "n", "v": "v", "a": "a", "r": "r"}
    return POS_MAP.get(wn_pos, None)


# LOAD SVM PACK (required for SVM method)
@st.cache_resource
def load_cpu_pack(pkl_path: str = "cpu_pack.pkl"):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Missing {pkl_path}.\n"
            f"Put your cpu_pack.pkl into artifacts/ and make sure it contains keys: svm_best, vec."
        )
    with open(pkl_path, "rb") as f:
        pack = pickle.load(f)

    if "svm_best" not in pack or "vec" not in pack:
        raise KeyError("cpu_pack.pkl must contain keys: svm_best, vec")

    return pack["svm_best"], pack["vec"]

# SVM FEATURIZE (window features)
def featurize(tokens: List[str], target_index: int, window: int = 3, wn_pos: Optional[str] = None) -> Dict[str, Any]:
    target_index = int(target_index)
    tokens = [str(t).lower() for t in tokens]

    t = tokens[target_index]
    left = tokens[max(0, target_index - window) : target_index]
    right = tokens[target_index + 1 : target_index + 1 + window]

    feat = {
        "target": t,
        "pos": wn_pos if wn_pos is not None else "__NOPOS__",
        "left_1": left[-1] if len(left) >= 1 else "__BOS__",
        "left_2": left[-2] if len(left) >= 2 else "__BOS__",
        "left_3": left[-3] if len(left) >= 3 else "__BOS__",
        "right_1": right[0] if len(right) >= 1 else "__EOS__",
        "right_2": right[1] if len(right) >= 2 else "__EOS__",
        "right_3": right[2] if len(right) >= 3 else "__EOS__",
    }
    return feat


def svm_wsd(sentence: str, target_index: int, wn_pos: Optional[str] = None) -> Optional[str]:
    toks = validate_input(sentence, target_index)
    feat = featurize(toks, target_index, wn_pos=wn_pos)
    svm_best, vec = load_cpu_pack()
    x = vec.transform([feat])
    return svm_best.predict(x)[0]

# LESK EXTENDED
STOPWORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()


def _clean_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return set(LEMM.lemmatize(w) for w in words)


def _extended_signature(synset) -> set:
    sig = _clean_words(synset.definition())
    for ex in synset.examples():
        sig |= _clean_words(ex)
    for s in synset.hypernyms() + synset.hyponyms():
        sig |= _clean_words(s.definition())
    return sig


def lesk_extended(sentence: str, target_index: int, wn_pos: Optional[str] = None) -> Optional[str]:
    toks = validate_input(sentence, target_index)
    target = toks[target_index].lower()

    wn_pos2 = normalize_pos(wn_pos)
    synsets = wn.synsets(target, pos=wn_pos2) if wn_pos2 else wn.synsets(target)
    if not synsets:
        return None

    context = set(
        LEMM.lemmatize(t.lower())
        for i, t in enumerate(toks)
        if i != target_index and t.isalpha() and t.lower() not in STOPWORDS
    )

    best_syn, best_score = None, 0
    for syn in synsets:
        score = len(context & _extended_signature(syn))
        if score > best_score:
            best_syn, best_score = syn, score

    return best_syn.name() if best_score > 0 else None

# BERT + COSINE (context vs gloss)
@st.cache_resource
def load_bert_backbone(model_name: str = "distilbert-base-uncased"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl, device


@st.cache_resource
def gloss_emb_cache() -> Dict[str, torch.Tensor]:
    return {}


def get_target_embedding(tokens: List[str], target_index: int, tokenizer, model, device) -> Optional[torch.Tensor]:
    sentence = " ".join(tokens)
    enc = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = enc.pop("offset_mapping")[0]
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]  # [seq, hidden]

    # char span for the target token in " ".join(tokens)
    char_start = sum(len(t) + 1 for t in tokens[:target_index])
    char_end = char_start + len(tokens[target_index])

    idxs = [i for i, (s, e) in enumerate(offsets.tolist()) if not (e <= char_start or s >= char_end)]
    if not idxs:
        return None

    return hidden[idxs].mean(dim=0)


def get_gloss_embedding(synset, tokenizer, model, device, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    key = synset.name()
    if key in cache:
        return cache[key]

    text = synset.definition()
    if synset.examples():
        text += " " + " ".join(synset.examples())

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]  # [seq, hidden]

    emb = hidden.mean(dim=0)
    cache[key] = emb
    return emb


def most_frequent_sense(lemma: str, wn_pos: Optional[str]) -> Optional[str]:
    synsets = wn.synsets(lemma, pos=wn_pos) if wn_pos else wn.synsets(lemma)
    return synsets[0].name() if synsets else None


def bert_wsd(sentence: str, target_index: int, wn_pos: Optional[str] = None, margin: float = 0.05) -> Optional[str]:
    toks = validate_input(sentence, target_index)
    wn_pos2 = normalize_pos(wn_pos)

    lemma = toks[target_index].lower()
    synsets = wn.synsets(lemma, pos=wn_pos2) if wn_pos2 else wn.synsets(lemma)
    if not synsets:
        return None

    tok, mdl, device = load_bert_backbone()
    cache = gloss_emb_cache()

    ctx_emb = get_target_embedding(toks, target_index, tok, mdl, device)
    if ctx_emb is None:
        return None

    scored: List[Tuple[str, float]] = []
    for syn in synsets:
        gloss_emb = get_gloss_embedding(syn, tok, mdl, device, cache)
        score = F.cosine_similarity(ctx_emb, gloss_emb, dim=0).item()
        scored.append((syn.name(), score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # tie-break by MFS if too close
    if len(scored) > 1 and (scored[0][1] - scored[1][1]) < margin:
        return most_frequent_sense(lemma, wn_pos2)

    return scored[0][0]


def compare_3(sentence: str, target_index: int, wn_pos: Optional[str] = None) -> Dict[str, Optional[str]]:
    return {
        "Lesk": lesk_extended(sentence, target_index, wn_pos),
        "SVM": svm_wsd(sentence, target_index, wn_pos),
        "BERT": bert_wsd(sentence, target_index, wn_pos),
    }


def synset_details(synset_name: Optional[str]) -> Optional[Dict[str, str]]:
    if not synset_name:
        return None
    try:
        s = wn.synset(synset_name)
        return {
            "synset": synset_name,
            "definition": s.definition(),
            "examples": "; ".join(s.examples()) if s.examples() else "(no examples)",
        }
    except Exception:
        return {"synset": synset_name, "definition": "(cannot load details)", "examples": "-"}

# SIDEBAR INPUT
with st.sidebar:
    st.header("🧩 Input")

    sentence = st.text_area(
        "Sentence",
        value="Hardly had Mrs. Roebuck driven off when a rusty pick-up truck came screeching to a stop.",
        height=120,
        help="Type any English sentence. Punctuation is kept as tokens to help indexing.",
    )

    tokens_preview = simple_tokenize_like_dataset(sentence) if sentence.strip() else []
    if tokens_preview:
        st.caption(
            "Token index helper:\n\n"
            + " | ".join([f"{i}:{t}" for i, t in enumerate(tokens_preview)])
        )
    else:
        st.caption("Type a sentence to see token indices.")

    max_idx = max(len(tokens_preview) - 1, 0)
    default_idx = min(12, max_idx)

    target_index = st.number_input(
        "target_index (0-based)",
        min_value=0,
        max_value=max_idx,
        value=default_idx,
        step=1,
        help="Index of the target word in the token list above.",
    )

    wn_pos = st.selectbox(
        "WordNet POS (optional)",
        ["", "n", "v", "a", "r"],
        index=0,
        help="n=noun, v=verb, a=adj, r=adv. Leave blank for all POS.",
    )
    wn_pos = wn_pos if wn_pos != "" else None

    st.divider()

    show_details = st.toggle("Show WordNet details", value=True)
    show_debug = st.toggle("Show debug panel", value=False)

    run_btn = st.button("🚀 Run WSD", type="primary", use_container_width=True)

# MAIN AREA
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Your input")
    st.markdown(
        f"""
<div class="card">
  <div class="kv">
    <div class="k">Sentence</div><div class="v">{sentence if sentence.strip() else "<i>(empty)</i>"}</div>
    <div class="k">target_index</div><div class="v">{int(target_index)}</div>
    <div class="k">wn_pos</div><div class="v">{wn_pos if wn_pos else "<i>(any)</i>"}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


with right:
    st.subheader("Status")
    # quick environment hints
    device_hint = "CUDA ✅" if torch.cuda.is_available() else "CPU 🧊"
    pack_hint = "OK ✅" if os.path.exists("artifacts/cpu_pack.pkl") else "Missing ❌"
    st.markdown(
        f"""
<div class="card">
  <div class="kv">
    <div class="k">BERT device</div><div class="v">{device_hint}</div>
    <div class="k">SVM pack</div><div class="v">{pack_hint}</div>
    <div class="k">Note</div><div class="v">First run may be slower (model download/cache).</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

# =========================
# RUN
# =========================
if run_btn:
    try:
        LOGGER.info("Run | idx=%s | pos=%s | sent=%s", int(target_index), wn_pos, sentence)

        with st.spinner("Running WSD..."):
            out = compare_3(sentence, int(target_index), wn_pos)

        # Results cards
        c1, c2, c3 = st.columns(3)

        def render_result_card(col, title: str, badge: str, syn: Optional[str]):
            d = synset_details(syn) if show_details else None
            syn_txt = syn if syn else "None"
            col.markdown(
                f"""
<div class="card">
  <div class="card-title">{title} <span class="badge">{badge}</span></div>
  <div class="mono">{syn_txt}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if show_details and d and syn:
                with col.expander("Details", expanded=False):
                    st.write("**Definition:**", d["definition"])
                    st.write("**Examples:**", d["examples"])

        render_result_card(c1, "Lesk", "extended", out["Lesk"])
        render_result_card(c2, "SVM", "linear", out["SVM"])
        render_result_card(c3, "BERT", "cosine", out["BERT"])

        st.markdown("<hr/>", unsafe_allow_html=True)

        # friendly highlight
        toks = validate_input(sentence, int(target_index))
        tgt = toks[int(target_index)]
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">🎯 Target word</div>
  <div class="mono">{tgt}</div>
  <div class="small">Tip: use the sidebar token index helper if you change the sentence.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if show_debug:
            st.write("")
            st.subheader("Debug panel")
            st.write("Tokenized:", toks)

    except Exception as e:
        LOGGER.exception("Error")
        st.error(str(e))

else:
    st.info("Enter your query in the sidebar → select target_index → ​​press **Run WSD**. (Don't select an index outside the token list, the app will 'read' it 😌)")
