"""Runway Archive: semantic search over existing Vogue Runway metadata."""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import hashlib
import html
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st

ROOT = Path(__file__).resolve().parent
DATASET_ID = "ROSCOSMOS/Vogue-Runway-Weburl-Dataset"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 3000

st.set_page_config(page_title="Runway Archive | Semantic Search", page_icon="👗", layout="wide")
st.markdown("""<style>
.stApp {background:#faf8f5;color:#24211f;}
.block-container {max-width:1240px;padding-top:2.6rem;}
h1,h2,h3 {font-family:Georgia,serif!important;font-weight:400!important;}
h1 {font-size:4.3rem!important;letter-spacing:-.055em;line-height:1.05!important;}
.eyebrow {font-size:11px;letter-spacing:.22em;color:#8c534d;margin-bottom:20px;}
.intro {font-size:18px;max-width:690px;color:#6c635c;line-height:1.65;margin:18px 0 28px;}
.look {background:white;border:1px solid #e9e2db;margin-bottom:24px;}
.look img {width:100%;height:330px;object-fit:contain;background:#f0ece7;display:block;}
.look-body {padding:18px 18px 20px;}
.look-brand {font:25px Georgia,serif;margin:4px 0 8px;}
.look-meta {font-size:12px;color:#82766c;letter-spacing:.025em;}
.look-tags {font-size:13px;color:#625b55;line-height:1.6;margin:12px 0;min-height:42px;}
.look-source {font-size:12px;color:#8c534d!important;}
.look-score {font-size:11px;color:#81776d;margin-top:12px;}
div[data-testid="stMetricValue"] {font-family:Georgia,serif;}
div.stButton>button {border-radius:2px;border:1px solid #d7ccc1;}
div.stFormSubmitButton>button {background:#302824;color:white;border:0;border-radius:2px;}
a {color:#8c534d;}
hr {border-color:#ded5cc;}
</style>""", unsafe_allow_html=True)


def clean_record(raw):
    """Keep source values; decode tags; skip rows without a valid source URL."""
    url = str(raw.get("url") or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "assets.vogue.com":
        return None
    tags = raw.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (TypeError, json.JSONDecodeError):
            tags = []
    if not isinstance(tags, list):
        tags = []
    tags = list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
    record = {field: str(raw.get(field) or "").strip() for field in
              ("key", "designer", "season", "year", "category")}
    record.update(url=url, tags=tags)
    return record


def make_search_text(record):
    """One short runway record is one chunk. No invented visual descriptions."""
    parts = []
    for field in ["designer", "season", "year", "category"]:
        value = record.get(field)
        if value is not None and str(value).strip():
            parts.append(f"{field.capitalize()}: {value}")
    tags = record.get("tags") or []
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    return ". ".join(parts)


@st.cache_data(show_spinner=False)
def load_records():
    """Use a bounded source sample, with a real 100-row offline fallback."""
    fallback = False
    try:
        from datasets import load_dataset
        stream = load_dataset(DATASET_ID, split="train", streaming=True)
        raw_rows = list(stream.take(SAMPLE_SIZE))
        if not raw_rows:
            raise ValueError("The dataset returned no records")
    except Exception:
        logging.exception("Dataset download failed; using bundled source preview")
        raw_rows = json.loads((ROOT / "sample_records.json").read_text())
        fallback = True
    records, seen = [], set()
    for raw in raw_rows:
        record = clean_record(raw)
        if record and record["url"] not in seen:
            seen.add(record["url"])
            records.append(record)
    if not records:
        raise ValueError("No usable runway records were found")
    return records, fallback


@st.cache_resource(show_spinner=False)
def build_search_index(records_json):
    """Load the model once and persist explicit embeddings in Chroma."""
    import chromadb
    import torch
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    model = SentenceTransformer(MODEL_ID, device="cpu")
    rows = json.loads(records_json)
    texts = [make_search_text(row) for row in rows]
    fingerprint = hashlib.sha256((MODEL_ID + records_json).encode()).hexdigest()[:16]
    client = chromadb.PersistentClient(
        path=str(ROOT / ".cache" / "chromadb"),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="runway_" + fingerprint,
        metadata={"hnsw:space": "cosine", "model": MODEL_ID},
        embedding_function=None,
    )
    if collection.count() != len(rows):
        for start in range(0, len(rows), 128):
            batch_rows, batch_texts = rows[start:start + 128], texts[start:start + 128]
            embeddings = model.encode(batch_texts, normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=False)
            collection.upsert(
                ids=[str(i) for i in range(start, start + len(batch_rows))],
                documents=batch_texts, embeddings=embeddings.tolist(),
                metadatas=[{key: row[key] or "Unknown" for key in
                            ("designer", "season", "year", "category")} for row in batch_rows],
            )
    return model, client, collection


def semantic_search(query, designer, season, category, limit):
    clauses = [{field: {"$eq": value}} for field, value in
               (("designer", designer), ("season", season), ("category", category))
               if value != "All"]
    where = {"$and": clauses} if len(clauses) > 1 else clauses[0] if clauses else None
    # Check the filtered count so zero matches return an honest empty result.
    eligible = len(collection.get(where=where, include=[])["ids"]) if where else collection.count()
    if eligible == 0:
        return []
    query_vector = model.encode([query], normalize_embeddings=True)
    result = collection.query(query_embeddings=query_vector.tolist(),
                              n_results=min(int(limit), eligible), where=where,
                              include=["distances"])
    return [(records[int(record_id)], 1.0 - float(distance))
            for record_id, distance in zip(result["ids"][0], result["distances"][0])]


def render_cards(matches):
    columns = st.columns(3)
    for index, (record, score) in enumerate(matches):
        esc = lambda value: html.escape(str(value), quote=True)
        meta = " · ".join(record[field] for field in ("season", "year", "category") if record[field])
        score_html = (f'<div class="look-score">Cosine similarity {score:.3f}</div>'
                      if score is not None else "")
        card = f'''<article class="look">
          <a href="{esc(record['url'])}" target="_blank" rel="noopener noreferrer">
            <img src="{esc(record['url'])}" alt="Runway photograph: {esc(record['designer'])}, {esc(meta)}"
                 loading="lazy" referrerpolicy="no-referrer"></a>
          <div class="look-body"><div class="look-meta">{esc(meta)}</div>
          <div class="look-brand">{esc(record['designer'] or 'Unknown designer')}</div>
          <div class="look-tags">{esc(', '.join(record['tags']))}</div>
          <a class="look-source" href="{esc(record['url'])}" target="_blank" rel="noopener noreferrer">View source image ↗</a>
          {score_html}</div></article>'''
        with columns[index % 3]:
            st.markdown(card, unsafe_allow_html=True)


st.markdown('<div class="eyebrow">AIPI 510 · FASHION / DATA / DISCOVERY</div>', unsafe_allow_html=True)
st.title("Runway Archive")
st.markdown('<div class="intro">A search through the runway, in your own words. Describe a garment or occasion to discover looks through their existing tags and collection details.</div>', unsafe_allow_html=True)

try:
    with st.spinner("Opening the archive and preparing its search index. The first visit can take a few minutes…"):
        records, fallback = load_records()
        model, client, collection = build_search_index(json.dumps(records, ensure_ascii=False, sort_keys=True))
except Exception:
    logging.exception("Search initialization failed")
    st.error("The archive could not start. Please reload in a minute. If it continues, the app owner should check the deployment logs.")
    st.stop()

if fallback:
    st.warning("The full sample is temporarily unavailable. Searching the bundled 100-record source preview instead.")
st.caption(f"{len(records):,} indexed photographs · Text embeddings · Search by meaning")

examples = ["A formal evening gown", "A warm coat or jacket", "A handbag or purse", "A tailored suit"]
for col, example in zip(st.columns(4), examples):
    if col.button(example, use_container_width=True):
        st.session_state["query"] = example
        st.session_state["auto_search"] = True

with st.form("runway_search"):
    query = st.text_input("Describe what you’re looking for", key="query",
                          placeholder="Try: a formal dress for an evening event")
    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    designer = c1.selectbox("Designer", ["All"] + sorted({r["designer"] for r in records if r["designer"]}))
    season = c2.selectbox("Season", ["All"] + sorted({r["season"] for r in records if r["season"]}))
    category = c3.selectbox("Collection", ["All"] + sorted({r["category"] for r in records if r["category"]}))
    limit = c4.selectbox("Results", [6, 9, 12], index=0)
    submitted = st.form_submit_button("Search the archive", use_container_width=True)

auto_search = st.session_state.pop("auto_search", False)
if submitted or auto_search:
    if not query.strip():
        st.warning("Enter a description to start your search.")
        st.session_state.pop("search_result", None)
    elif len(query) > 1000:
        st.warning("Please keep your description under 1,000 characters.")
        st.session_state.pop("search_result", None)
    else:
        with st.spinner("Finding related records…"):
            matches = semantic_search(query.strip(), designer, season, category, limit)
        st.session_state["search_result"] = (query.strip(), matches)

st.divider()
if "search_result" in st.session_state:
    searched_query, matches = st.session_state["search_result"]
    st.subheader("Your matches")
    st.write(f'Results for “{searched_query}”')
    if matches:
        if matches[0][1] < 0.25:
            st.info("These are the closest available records, but their similarity is low. Try a garment type or broader wording.")
        st.caption("Higher cosine similarity means closer text meaning. It is not a confidence percentage or a guarantee of visual relevance.")
        render_cards(matches)
    else:
        st.info("No records match these filters. Try selecting All for a wider search.")
else:
    st.subheader("From the archive")
    st.caption("A preview of indexed records. Enter a search above to rank looks by meaning.")
    render_cards([(record, None) for record in records[:6]])

with st.expander("How this search works"):
    st.markdown("""**1. Source:** Load up to the first 3,000 records from the Vogue Runway WebURL dataset. This bounded sample is not a random sample of the complete archive.

**2. Chunk:** Treat each photo's metadata as one short document. Join its designer, season, year, category and source tags. Missing fields are omitted; descriptions are not invented.

**3. Embed:** The pretrained all-MiniLM-L6-v2 model converts each document into a normalized vector of 384 numbers. No model training takes place.

**4. Store:** Save vectors and their document text in a local persistent Chroma database, with cosine distance. The host can reset local disk; the app rebuilds its index if needed.

**5. Search:** Embed your query with the same model, apply exact dropdown filters, then retrieve the closest vectors. Display the records' linked source photographs.

**Limits:** This searches text metadata, not image pixels. Tags can be broad or inaccurate; colors, fabrics and aesthetic moods may be absent. A nearest match can still be irrelevant. External image links can stop working. This is retrieval, with no generated answer and no RAG generation step.""")
    st.code('embeddings = model.encode(texts, normalize_embeddings=True)\nresults = collection.query(query_embeddings=query_vector.tolist(), n_results=6)', language="python")

st.divider()
st.caption("Educational project · Metadata: ROSCOSMOS / Vogue Runway WebURL Dataset · Images belong to Vogue Runway and their respective rights holders. No affiliation or endorsement. Source images are linked, not bundled with this app.")
st.markdown("[Dataset and provenance](https://huggingface.co/datasets/ROSCOSMOS/Vogue-Runway-Weburl-Dataset) · [Project code](https://github.com/snehal-jindal/runway-semantic-search)")
