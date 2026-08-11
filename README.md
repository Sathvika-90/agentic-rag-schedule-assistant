# Agentic RAG Schedule Assistant

A 30-day schedule assistant using:
- ChromaDB vector database
- Sentence-Transformers embeddings
- Basic RAG retrieval
- Two agent tools: `get_schedule` and `update_schedule`
- Streamlit UI

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model.

## Example requests
- What do I have scheduled tomorrow?
- Am I free Friday afternoon?
- Add a meeting on August 15 at 3 PM.
- Move my meeting from 2 PM to 4 PM.

## Deployment
Deploy `app.py`, `schedule.json`, and `requirements.txt` to Streamlit Community Cloud, Render, Railway, or another Python hosting service.
