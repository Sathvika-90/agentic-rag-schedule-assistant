
import json, re, uuid
from datetime import datetime, date, timedelta
from pathlib import Path

import chromadb
import streamlit as st
from dateutil import parser as dtparser
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DATA_FILE = BASE / "schedule.json"
CHROMA_DIR = BASE / "chroma_db"

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅", layout="wide")

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()

def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection("schedule_events")

def load_events():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))

def save_events(events):
    DATA_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")

def event_text(e):
    return f"{e['title']} | {e['type']} | {e['date']} | {e['start']}-{e['end']} | {e['description']}"

def rebuild_index():
    events = load_events()
    client = get_chroma_client()
    try:
        client.delete_collection("schedule_events")
    except Exception:
        pass
    col = client.get_or_create_collection("schedule_events")
    if events:
        ids = [e["id"] for e in events]
        docs = [event_text(e) for e in events]
        metas = [{k: str(v) for k, v in e.items()} for e in events]
        embeddings = get_embedder().encode(docs).tolist()
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

def ensure_index():
    col = get_collection()
    if col.count() != len(load_events()):
        rebuild_index()

def normalize_date(text):
    today = date.today()
    t = text.lower().strip()
    if "today" in t: return today
    if "tomorrow" in t: return today + timedelta(days=1)
    weekdays = {x.lower(): i for i, x in enumerate(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])}
    for name, idx in weekdays.items():
        if name in t:
            delta = (idx - today.weekday()) % 7
            if delta == 0 and "next" in t: delta = 7
            return today + timedelta(days=delta)
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:\s+(\d{4}))?", t)
    if m:
        year = int(m.group(3) or today.year)
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {year}", "%b %d %Y").date()
        except:
            try: return datetime.strptime(f"{m.group(1)} {m.group(2)} {year}", "%B %d %Y").date()
            except: pass
    return None

def parse_time(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text.lower())
    if not m: return None
    h, minute, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h != 12: h += 12
    if ap == "am" and h == 12: h = 0
    return h * 60 + minute

def get_schedule(query, target_date=None, target_time=None, top_k=6):
    ensure_index()
    col = get_collection()
    where = {"date": str(target_date)} if target_date else None
    emb = get_embedder().encode([query]).tolist()
    res = col.query(query_embeddings=emb, n_results=top_k, where=where)
    results = []
    for md in (res.get("metadatas") or [[]])[0]:
        results.append(dict(md))
    if target_time is not None:
        results.sort(key=lambda e: abs(int(e["start"][:2])*60 + int(e["start"][3:]) - target_time))
    return results

def update_schedule(action, title=None, event_date=None, start=None, end=None, event_type="meeting", description=""):
    events = load_events()
    action = action.lower()
    if action == "add":
        new = {
            "id": "evt-" + uuid.uuid4().hex[:8], "title": title or "Untitled",
            "type": event_type, "date": str(event_date),
            "start": start or "09:00", "end": end or start or "10:00",
            "description": description or ""
        }
        events.append(new)
        save_events(events); rebuild_index()
        return f"Added: {event_text(new)}"
    if action in ("remove", "delete"):
        matches = [e for e in events if (title or "").lower() in e["title"].lower()
                   and (not event_date or e["date"] == str(event_date))]
        if not matches: return "I couldn't find a matching event to remove."
        events = [e for e in events if e["id"] != matches[0]["id"]]
        save_events(events); rebuild_index()
        return f"Removed: {event_text(matches[0])}"
    if action == "update":
        matches = [e for e in events if (title or "").lower() in e["title"].lower()
                   and (not event_date or e["date"] == str(event_date))]
        if not matches: return "I couldn't find a matching event to update."
        e = matches[0]
        if start: e["start"] = start
        if end: e["end"] = end
        if event_date: e["date"] = str(event_date)
        if title and title.lower() not in e["title"].lower(): e["title"] = title
        save_events(events); rebuild_index()
        return f"Updated: {event_text(e)}"
    return "Unsupported update action."

def agent(user_query):
    q = user_query.lower()
    d = normalize_date(q)
    t = parse_time(q)

    # Tool decision: retrieval for questions, update_schedule for mutations.
    mutation = any(x in q for x in ["add ", "schedule ", "create ", "book ", "move ", "reschedule ", "change ", "update ", "remove ", "delete ", "cancel "])
    if mutation:
        # Move/reschedule: identify existing event by semantic retrieval, then apply new time.
        if any(x in q for x in ["move ", "reschedule ", "change "]):
            new_time = t
            if new_time is None: return "Please provide the new time."
            candidates = get_schedule(user_query, target_date=d, top_k=5)
            if not candidates: return "I couldn't find the meeting to move."
            chosen = candidates[0]
            old_start = int(chosen["start"][:2]) * 60 + int(chosen["start"][3:])
            old_end = int(chosen["end"][:2]) * 60 + int(chosen["end"][3:])
            duration = max(30, old_end - old_start)
            new_end = new_time + duration
            return update_schedule(
                "update",
                title=chosen["title"],
                event_date=chosen["date"],
                start=f"{new_time//60:02d}:{new_time%60:02d}",
                end=f"{new_end//60:02d}:{new_end%60:02d}",
                event_type=chosen["type"],
                description=chosen["description"]
            )
        # Add/create/book
        if any(x in q for x in ["add ", "create ", "book ", "schedule "]):
            if not d: return "Please include the date (for example, August 15)."
            title = re.sub(r"^(add|create|book|schedule)\s+(a\s+)?", "", user_query, flags=re.I)
            title = re.split(r"\s+on\s+|\s+at\s+", title, flags=re.I)[0].strip() or "New meeting"
            # Remove a trailing period from titles created from natural-language requests.
            title = title.rstrip(".")
            stime = f"{t//60:02d}:{t%60:02d}" if t is not None else "09:00"
            return update_schedule("add", title=title, event_date=d, start=stime,
                                   end=f"{(t+60)//60:02d}:{(t+60)%60:02d}" if t is not None else "10:00")
        if any(x in q for x in ["remove ", "delete ", "cancel "]):
            title = re.sub(r"^(remove|delete|cancel)\s+", "", user_query, flags=re.I).strip()
            return update_schedule("remove", title=title, event_date=d)
    # Retrieval tool
    results = get_schedule(user_query, target_date=d, target_time=t, top_k=8)
    if not results: return "I couldn't find any matching schedule entries."
    if "free" in q or "available" in q:
        return free_time_answer(results, d, q)
    lines = ["Here are the relevant schedule entries:"]
    for e in results:
        lines.append(f"• {e['date']} {e['start']}-{e['end']} — {e['title']} ({e['type']})")
    return "\n".join(lines)

def free_time_answer(results, d, q):
    if not d:
        d = normalize_date(q) or date.today()
    day_events = get_schedule("schedule", target_date=d, top_k=30)
    # Look at afternoon if requested; otherwise return overall availability.
    afternoon = "afternoon" in q
    start, end = (12*60, 17*60) if afternoon else (9*60, 18*60)
    busy = []
    for e in day_events:
        a = int(e["start"][:2])*60 + int(e["start"][3:])
        b = int(e["end"][:2])*60 + int(e["end"][3:])
        if b > start and a < end: busy.append((max(a,start), min(b,end), e["title"]))
    if not busy: return f"Yes — you are free during the requested period on {d.strftime('%A, %B %d')}."
    busy.sort()
    slots = []
    cur = start
    for a,b,title in busy:
        if a > cur: slots.append((cur,a))
        cur = max(cur,b)
    if cur < end: slots.append((cur,end))
    fmt = lambda x: datetime.strptime(f"{x//60:02d}:{x%60:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
    free = ", ".join(f"{fmt(a)}–{fmt(b)}" for a,b in slots)
    busy_s = ", ".join(f"{title} ({fmt(a)}–{fmt(b)})" for a,b,title in busy)
    return f"Not completely free. Busy: {busy_s}. Free windows: {free or 'none'}."

ensure_index()

st.title("📅 Agentic RAG Schedule Assistant")
st.caption("30-day schedule • ChromaDB vector retrieval • two agent tools: get_schedule and update_schedule")

with st.sidebar:
    st.header("Schedule")
    events = sorted(load_events(), key=lambda e: (e["date"], e["start"]))
    st.write(f"{len(events)} events in the sample 30-day schedule.")
    if st.button("Rebuild ChromaDB index"):
        rebuild_index(); st.success("Index rebuilt.")
    st.download_button("Download schedule.json", json.dumps(events, indent=2), "schedule.json", "application/json")

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Ask the agent")
    examples = [
        "What do I have scheduled tomorrow?",
        "Am I free Friday afternoon?",
        "Add a meeting on August 15 at 3 PM.",
        "Move my meeting from 2 PM to 4 PM.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex
    query = st.text_input("Your request", value=st.session_state.get("query", ""), placeholder="Ask about your schedule or change it...")
    if st.button("Run agent", type="primary") and query:
        with st.spinner("Agent deciding which tool to use..."):
            st.markdown("### Agent response")
            st.write(agent(query))

with right:
    st.subheader("Upcoming events")
    for e in events[:12]:
        st.markdown(f"**{e['date']} · {e['start']}–{e['end']}**  \n{e['title']} · {e['type']}")
