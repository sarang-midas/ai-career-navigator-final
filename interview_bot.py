
import streamlit as st
from typing import List

from llm_client import get_openai_client

SYSTEM_PROMPT = "You are a helpful, strict mock interviewer. Ask one question at a time, wait for answer, then give brief feedback and the next question."

# Simple offline mock questions to use when OpenAI isn't configured
_OFFLINE_QUESTIONS: List[str] = [
    "Tell me about yourself.",
    "Why do you want this role?",
    "Describe a challenging project you worked on and how you solved problems.",
    "Explain a technical concept (e.g., normalization) to a non-technical audience.",
]


def run_mock_interview(target_role: str = "Data Analyst"):
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Start a mock interview for the role: {target_role}. Begin with a friendly greeting and the first HR question."},
        ]
        st.session_state._offline_q_idx = 0

    st.write(f"**Target Role:** {target_role}")
    container = st.container()

    # Display history excluding system
    for msg in st.session_state.interview_history:
        if msg["role"] == "assistant":
            container.chat_message("assistant").markdown(msg["content"])
        elif msg["role"] == "user" and msg["content"] != st.session_state.interview_history[1]["content"]:
            container.chat_message("user").markdown(msg["content"])

    user_inp = st.chat_input("Type your answer and press Enter")
    if user_inp:
        st.session_state.interview_history.append({"role": "user", "content": user_inp})
        # Try to use OpenAI; if unavailable, use simple offline feedback/questions
        openai_client = get_openai_client()
        if openai_client is None:
            # Offline: give brief canned feedback and ask next question
            idx = getattr(st.session_state, "_offline_q_idx", 0)
            feedback = "Good structure — try to add a concrete metric next time."
            st.session_state.interview_history.append({"role": "assistant", "content": feedback})
            next_q = _OFFLINE_QUESTIONS[min(idx + 1, len(_OFFLINE_QUESTIONS) - 1)]
            st.session_state.interview_history.append({"role": "assistant", "content": next_q})
            st.session_state._offline_q_idx = idx + 1
            st.rerun()

        with st.spinner("Evaluating..."):
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.interview_history,
                temperature=0.5,
            )
        answer = resp.choices[0].message.content
        st.session_state.interview_history.append({"role": "assistant", "content": answer})
        st.rerun()
