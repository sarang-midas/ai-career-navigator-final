import os
import pandas as pd
import streamlit as st

# Use centralized LLM client helper (Groq only)
from llm_client import get_groq_client

# Default Groq model
_MODEL = "llama-3.1-8b-instant"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "skills_dataset.csv")


def load_role_skills():
    try:
        df = pd.read_csv(DATA_PATH)
        df['role'] = df['role'].str.strip().str.lower()
        df['skills'] = df['skills'].fillna('').astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["role", "skills"])


def _chat(prompt: str, model: str = None) -> str:
    """
    Use Groq client if available. If not, return an offline fallback response.
    """
    model = model or _MODEL

    # ---------------------------
    # ⭐ GROQ CALL
    # ---------------------------
    groq_client = get_groq_client()
    if groq_client is not None:
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert AI career mentor for students."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content
        except Exception as e:
            try:
                st.error(f"Groq API request failed: {e}")
            except Exception:
                pass

    # ---------------------------
    # ⭐ FALLBACK (OFFLINE MODE)
    # ---------------------------
    header = "**(Fallback) Career Advisor — offline mode**\n\n"
    body = (
        "Groq API not available. Showing a sample response so the UI stays usable.\n\n"
    )
    example = (
        "- Suggested roles: Data Analyst, ML Engineer, BI Analyst\n"
        "- Must-have skills: Python, SQL, Statistics, Excel\n"
        "- Beginner projects: Sales dashboard, Kaggle EDA, simple ML model\n"
        "- Expected salary: 3L–7L (entry-level, India)\n"
    )
    return header + body + example


def get_career_paths(skills: str, interests: str, education: str, experience: str) -> str:
    prompt = f"""
    Profile:
    - Education: {education}
    - Experience: {experience}
    - Skills: {skills}
    - Interests: {interests}

    Task: Suggest 4–6 high-demand career paths suitable for this profile in India:
    - What the role does
    - Why it's a good fit
    - 3 must-have skills
    - Starter projects
    - Entry-level salary (INR)
    Format in clean markdown.
    """
    return _chat(prompt)


def analyze_skill_gaps(user_skills_csv: str, target_role: str, role_skills_df: pd.DataFrame):
    user = {s.strip().lower() for s in user_skills_csv.split(",") if s.strip()}
    role = (target_role or "data analyst").strip().lower()

    rows = role_skills_df[role_skills_df['role'] == role]
    required = set()

    for _, r in rows.iterrows():
        required.update({x.strip().lower() for x in str(r['skills']).split(",") if x.strip()})

    if not required:
        required = {
            "python", "sql", "statistics", "excel",
            "data visualization", "tableau", "power bi", "etl"
        }

    gaps = sorted(list(required - user))
    matches = sorted(list(required & user))
    coverage = int((len(matches) / len(required)) * 100) if required else 0

    return {
        "target_role": role,
        "have_skills": matches,
        "missing_skills": gaps,
        "coverage_percent": coverage
    }


def get_learning_plan(skills: str, interests: str, duration: str, target_role: str) -> str:
    prompt = f"""
    Create a structured learning plan to become a strong {target_role or 'Data Analyst'} in {duration}.
    Student’s current skills: {skills}
    Interests: {interests}

    Include:
    - Phase-by-phase roadmap
    - Weekly outcomes
    - Free resources (YouTube, docs, projects)
    - 3 portfolio projects with acceptance criteria
    - Final checklist

    Keep it practical and India-friendly.
    """
    return _chat(prompt)
