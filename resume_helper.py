from llm_client import get_openai_client


def _offline_resume_tips(resume_text: str, target_role: str) -> str:
    """Return simple, deterministic resume improvement tips when no LLM is available."""
    tips = []
    text = (resume_text or "").lower()
    tips.append("## Offline resume tips (no OpenAI key available)")
    tips.append("- Use strong action verbs (e.g., 'Implemented', 'Led', 'Reduced').")
    tips.append("- Quantify achievements where possible (e.g., 'improved accuracy by 12%').")
    tips.append("- Keep bullets short and results-focused.")
    role_words = [w for w in (target_role or "").lower().split() if w]
    missing = [w for w in role_words if w not in text]
    if missing:
        tips.append(f"- Add role keywords: {', '.join(missing)}")
    else:
        tips.append("- Resume contains target-role keywords — good.")
    tips.append("- Put key skills and projects near the top; keep to one page for entry-level.")
    return "\n".join(tips)


def improve_resume(resume_text: str, target_role: str = "Data Analyst") -> str:
    """Improve resume using OpenAI when available; otherwise provide offline tips.

    This function is safe to import even when OPENAI_API_KEY is not set.
    """
    client = get_openai_client()
    if client is None:
        return _offline_resume_tips(resume_text, target_role)

    prompt = f"""
You are an ATS resume expert.
Improve the following resume for the role: {target_role}.
1) Give an ATS optimization checklist.
2) Rewrite the summary, skills, and 2 sample bullet points with strong verbs + metrics.
3) Suggest keywords and a clean one-page layout.
Resume text:
---
{resume_text}
---
Format output in markdown.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Be precise, practical, and recruiter-friendly."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        # handle response shape defensively
        if resp and getattr(resp, "choices", None):
            choice = resp.choices[0]
            if hasattr(choice, "message") and isinstance(choice.message, dict):
                return choice.message.get("content", "").strip()
            try:
                return choice.message.content.strip()
            except Exception:
                pass
        return "(No content returned from OpenAI)"
    except Exception:
        return _offline_resume_tips(resume_text, target_role)
        return _offline_resume_tips(resume_text, target_role)

