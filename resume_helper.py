"""
Resume improvement helper using Groq ONLY.
Provides:
- improve_resume(): main resume enhancer (Groq → fallback)
- _offline_resume_tips(): deterministic fallback when LLM unavailable
"""

from llm_client import get_groq_client


# -----------------------------
# 🔹 Fallback (Offline Mode)
# -----------------------------
def _offline_resume_tips(resume_text: str, target_role: str) -> str:
    """Return simple, deterministic resume tips when no Groq model is available."""
    text = (resume_text or "").lower()
    tips = [
        "## Offline Resume Tips (LLM not available)",
        "- Use strong action verbs (Implemented, Designed, Automated, Delivered).",
        "- Quantify achievements (e.g., 'improved accuracy by 15%').",
        "- Keep bullets short and results-focused.",
        "- Place skills + summary at the top – ATS prefers structured format.",
    ]

    # Add missing keywords for the target role
    role_words = [w for w in (target_role or "").lower().split() if w]
    missing = [w for w in role_words if w not in text]
    if missing:
        tips.append(f"- Add keywords for this role: {', '.join(missing)}")
    else:
        tips.append("- Good: Resume already includes target role keywords.")

    tips.append("- Keep resume within one page for entry-level roles.")

    return "\n".join(tips)


# -----------------------------
# 🔹 Main Function (Groq Only)
# -----------------------------
def improve_resume(resume_text: str, target_role: str = "Data Analyst") -> str:
    """
    Improve resume using Groq LLM.
    If Groq API not available → return offline fallback tips.
    """
    client = get_groq_client()
    if client is None:
        return _offline_resume_tips(resume_text, target_role)

    prompt = f"""
You are an ATS resume optimization expert.

Improve the following resume for the role: **{target_role}**.

### Your Tasks:
1. Provide an ATS optimization checklist.
2. Rewrite:
   - Resume summary
   - Skills section
   - Two bullet points (strong action verbs + metrics)
3. Suggest important keywords for this role.
4. Provide a clean, one-page resume structure.

### Resume:
---
{resume_text}
---

Format everything in clean markdown.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional ATS resume reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return _offline_resume_tips(resume_text, target_role)
