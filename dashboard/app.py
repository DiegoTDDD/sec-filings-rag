"""Streamlit dashboard for the SEC Filings RAG system.

A thin, professional UI over the FastAPI backend: the user picks an optional
company filter, types (or clicks) a question, and gets a grounded answer with the
exact filing sections it was drawn from. All heavy lifting (retrieval, re-ranking,
generation) happens in the API; this file only handles presentation and calls
POST /ask.

Run (with the API already up on :8000):
    streamlit run dashboard/app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

COMPANIES = {
    "All companies": None,
    "Apple (AAPL)": "AAPL",
    "NVIDIA (NVDA)": "NVDA",
    "JPMorgan (JPM)": "JPM",
}

EXAMPLE_QUESTIONS = [
    "How does NVIDIA describe the risk from U.S. export controls to China?",
    "What supply chain risks does Apple disclose?",
    "What regulatory and legal risks does JPMorgan describe?",
    "How did NVIDIA's risk factors change year over year?",
]

st.set_page_config(page_title="SEC Filings RAG", page_icon="📄", layout="centered")

st.title("SEC Filings RAG")
st.caption(
    "Ask questions about the 10-K and 10-Q filings of Apple, NVIDIA and JPMorgan. "
    "Answers are generated only from the filings and cite the exact sections used."
)

# --- Controls -------------------------------------------------------------

company_label = st.selectbox("Company filter", list(COMPANIES.keys()))
ticker = COMPANIES[company_label]

st.write("**Example questions** (click to use):")
cols = st.columns(2)
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
        st.session_state["question"] = q

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    height=90,
    placeholder="e.g. What are the main competitive risks disclosed?",
)

ask = st.button("Ask", type="primary", use_container_width=True)

# --- Query + results ------------------------------------------------------

if ask and question.strip():
    with st.spinner("Retrieving filings and generating an answer..."):
        try:
            resp = requests.post(
                API_URL, json={"question": question, "ticker": ticker}, timeout=180
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            st.error(
                "Could not reach the API. Make sure it is running on port 8000.\n\n"
                f"Details: {exc}"
            )
            st.stop()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    for i, s in enumerate(data["sources"], 1):
        st.markdown(
            f"**[{i}]** {s['ticker']} · {s['form']} · filed {s['filing_date']} · "
            f"*{s['section'].replace('_', ' ')}*"
        )

elif ask:
    st.warning("Please enter a question first.")
