# Runway Archive

[Open the public app](https://snehal-runway-archive.streamlit.app/)

A semantic fashion search app for AIPI 510. Describe a garment or occasion and retrieve runway records by the meaning of their existing text metadata.

## Run

Use Python 3.11, install `requirements.txt`, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy this public repository with Streamlit Community Cloud, branch `main`, entrypoint `app.py`, Python 3.11. No API keys are needed. First startup downloads the pretrained embedding model and dataset sample and builds the index; subsequent searches reuse the cached model and index.

## Pipeline

1. Stream up to the first 3,000 rows from [ROSCOSMOS/Vogue-Runway-Weburl-Dataset](https://huggingface.co/datasets/ROSCOSMOS/Vogue-Runway-Weburl-Dataset). Keep unique valid Vogue source URLs. This is a bounded convenience sample, not a random sample of the full archive.
2. Treat one record as one semantic chunk. Join its existing designer, season, year, category, and tags. Omit missing fields. No visual descriptions are invented.
3. Encode documents with `sentence-transformers/all-MiniLM-L6-v2`: 384-dimensional normalized text embeddings. Use the same model for queries.
4. Store explicit embeddings, documents, IDs, and filter metadata in Chroma `PersistentClient`, using cosine distance. The collection name includes a fingerprint of the dataset and model configuration. Reopening an existing complete collection avoids re-encoding.
5. Apply exact optional designer/season/category filters and retrieve the closest vectors. Show `1 - cosine distance` as similarity, never as a confidence probability.
6. Render a Streamlit gallery with source photo links, collection metadata, and original tags.

`sample_records.json` contains 100 real records extracted from the dataset's public preview. It provides a clearly labeled fallback when the larger sample cannot be downloaded. It contains metadata and URLs, not image files.

## Persistence and limitations

Chroma writes to `.cache/chromadb` on the host. Free hosting may reset local disk or sleep; the app rebuilds its index on startup when necessary. The model remains pretrained: there is no training or fine-tuning. This app retrieves existing records; it does not generate answers.

Search reads tags and collection information, not image pixels. Some source tags are broad, noisy, or refer to people/backgrounds. Colors, materials, and aesthetic moods may not be represented. Nearest matches can be irrelevant; low similarity is indicated. Dropdown filters are exact; natural language negation and numerical constraints are not guaranteed. Image links are external and can fail.

## Source attribution

The source dataset is a third-party URL and metadata index. Images remain associated with Vogue Runway and their respective photographers and rights holders; this project does not claim rights to them. Source images are linked directly, not downloaded into or redistributed with the repository. No affiliation or endorsement is implied. See the dataset card for provenance and rights information. The repository does not grant a license for the underlying images.
