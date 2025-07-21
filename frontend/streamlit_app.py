# streamlit_app.py

import streamlit as st
import requests

# Page Config
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title & Mode Toggle
st.title("🏥 MedGuide-AI : Healthcare Conversational Assistant")
qa_mode = st.radio("Select Mode:", ["Extractive QA", "Generative QA"], horizontal=True)
st.markdown(
    """
Ask questions about medical symptoms and insurance policies.  
Powered by a Retrieval-Augmented RoBERTa-SQuAD2 pipeline with safety guardrails.  
"""
)

# Session State for Chat History 
if "history" not in st.session_state:
    st.session_state.history = []
    
if "latest" not in st.session_state:
    st.session_state.latest = None 
    
if "history_display_count" not in st.session_state:
    st.session_state.history_display_count = 3  
    
# Generative Placeholder
if qa_mode == "Generative QA":
    st.warning("🧪 Generative QA is under development and will be available soon. Please switch to 'Extractive QA'.")
    st.stop()

# Input Form 
with st.form("qa_form", clear_on_submit=True):
    user_question = st.text_input("Your question:", placeholder="e.g. What are the signs of stroke?", key="user_input")
    submitted = st.form_submit_button("Send")

# Handle Submission 
if submitted and user_question:
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"question": user_question, "history": [q for q, _ in st.session_state.history]}
        )
        data = response.json()
        answer = data.get("answer", "Error: No answer returned.")
        sources = data.get("sources", [])
        
        st.session_state.latest = (user_question, answer, sources)
        st.session_state.history.append((user_question, answer))
        
    except Exception as e:
        st.session_state.latest = (user_question, f"Error contacting API: {e}", [])
    
# Display Latest Q&A 
if st.session_state.latest:
    q, a, sources = st.session_state.latest
    st.subheader("Latest Answer")
    st.markdown(f"**🧠 You:** {q}")
    st.markdown(f"**🤖 Assistant:** {a}")
    if sources:
        st.markdown("**📚 Sources:**")
        for src in sorted(set(sources)):
            st.markdown(f"- {src}")
    st.markdown("---")

# Display Chat History 
if len(st.session_state.history) > 1:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.subheader("📜 Previous Q&A")
    with col2:
        st.markdown("")  
        if st.button("🧹 Clear Chat History"):
            st.session_state.history = [st.session_state.history[-1]]
            st.session_state.history_display_count = 3
            st.rerun()

    # Initialize display count if not set
    if "history_display_count" not in st.session_state:
        st.session_state.history_display_count = 3

    total_prev = len(st.session_state.history) - 1  # Exclude latest
    count = st.session_state.history_display_count

    for q, a in reversed(st.session_state.history[-(count + 1):-1]):
        st.markdown(f"**🧠 You:** {q}")
        st.markdown(f"**🤖 Assistant:** {a}")
        st.markdown("---")

    # Show button to load more if applicable
    if total_prev > count:
        if st.button("View previous interactions"):
            st.session_state.history_display_count += 3
            st.rerun()