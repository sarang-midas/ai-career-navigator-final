import streamlit as st
from typing import List
from llm_client import get_groq_client

SYSTEM_PROMPT = (
    "You are a strict but helpful mock interviewer. "
    "Ask one question at a time. After the candidate answers, "
    "give brief feedback and then ask the next question."
)

# Offline fallback questions
_OFFLINE_QUESTIONS: List[str] = [
    "Tell me about yourself.",
    "Why do you want this role?",
    "Describe a challenging project you worked on and how you solved it.",
    "Explain a technical concept to a non-technical person.",
]


def run_mock_interview(target_role: str = "Data Analyst"):
    """
    Main mock-interview function using Groq.
    If Groq is unavailable -> use offline canned questions.
    """
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Start a mock interview for the role: {target_role}. "
                    "Begin with a friendly greeting and the first HR question."
                ),
            },
        ]
        st.session_state._offline_q_idx = 0

    st.write(f"**🎯 Target Role:** {target_role}")
    chat_box = st.container()

    # Display conversation
    for msg in st.session_state.interview_history[1:]:
        if msg["role"] == "assistant":
            chat_box.chat_message("assistant").markdown(msg["content"])
        elif msg["role"] == "user":
            chat_box.chat_message("user").markdown(msg["content"])

    # User response
    user_input = st.chat_input("Answer the question...")
    if not user_input:
        return

    # Add user input to history
    st.session_state.interview_history.append({"role": "user", "content": user_input})

    # -------------------------------
    # TRY GROQ LLM
    # -------------------------------
    groq_client = get_groq_client()

    if groq_client is None:
        # -------------------------------
        # OFFLINE FALLBACK MODE
        # -------------------------------
        idx = st.session_state._offline_q_idx

        feedback = "Good response. Try to add measurable results next time."
        st.session_state.interview_history.append({"role": "assistant", "content": feedback})

        next_q = _OFFLINE_QUESTIONS[min(idx + 1, len(_OFFLINE_QUESTIONS) - 1)]
        st.session_state.interview_history.append({"role": "assistant", "content": next_q})
        st.session_state._offline_q_idx = idx + 1

        st.rerun()
        return

    # -------------------------------
    # ONLINE MODE — GROQ RESPONSE
    # -------------------------------
    with st.spinner("Mock interviewer is reviewing your answer..."):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.interview_history,
                temperature=0.4,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"(Groq failed — fallback mode)\nError: {e}"
            st.error(str(e))

    st.session_state.interview_history.append({"role": "assistant", "content": answer})

    st.rerun()
