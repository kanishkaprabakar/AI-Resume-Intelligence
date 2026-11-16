
import os
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_reader import extract_resume_text
from utils.skill_extractor import extract_skills


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 2. PROFESSIONAL DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #18294a 0%, transparent 35%),
            linear-gradient(135deg, #080f1e 0%, #0d1729 55%, #101c32 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: #0a1324;
        border-right: 1px solid #25344d;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        color: #cbd5e1;
    }

    .hero {
        padding: 30px;
        border: 1px solid #334766;
        border-radius: 18px;
        background: linear-gradient(120deg, #172b4d, #12213a);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #c1d1e8;
        font-size: 15px;
    }

    .section-card {
        background: #111e32;
        border: 1px solid #2b3d59;
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
    }

    .metric-card {
        background: #14233a;
        border: 1px solid #344966;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-label {
        color: #a8bad3;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
    }

    .skill-tag {
        display: inline-block;
        background: #173b38;
        color: #86efac;
        border: 1px solid #26745e;
        border-radius: 20px;
        padding: 7px 12px;
        margin: 4px;
        font-size: 13px;
    }

    .missing-tag {
        display: inline-block;
        background: #3a2630;
        color: #fda4af;
        border: 1px solid #854354;
        border-radius: 20px;
        padding: 7px 12px;
        margin: 4px;
        font-size: 13px;
    }

    textarea, input {
        background-color: #17263d !important;
        color: #ffffff !important;
        border: 1px solid #425675 !important;
        border-radius: 9px !important;
    }

    textarea::placeholder, input::placeholder {
        color: #b2c0d4 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] {
        background: #14233a;
        border: 1px dashed #58708f;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stFileUploader"] * {
        color: #e2e8f0 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 11px 22px;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        border: none;
        color: white;
    }

    [data-testid="stMetric"] {
        background: #14233a;
        border: 1px solid #344966;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #b8c8df !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    hr {
        border-color: #30415b;
    }

    .small-note {
        color: #a8bad3;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. SKILL NORMALIZATION
# --------------------------------------------------

def normalize_skills(skills):
    aliases = {
        "llm": "large language models",
        "llms": "large language models",
        "rag": "retrieval augmented generation",
        "dsa": "data structures and algorithms",
        "algorithms": "data structures and algorithms",
        "data structures": "data structures and algorithms",
        "rest api": "api",
        "rest apis": "api",
        "apis": "api",
        "vector database": "vector databases"
    }

    normalized = set()

    for skill in skills:
        skill = skill.lower().strip()
        normalized.add(aliases.get(skill, skill))

    return sorted(normalized)


# --------------------------------------------------
# 4. TEXT SIMILARITY
# --------------------------------------------------

def calculate_similarity(resume_text, job_text):

    if not resume_text.strip() or not job_text.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")

        vectors = vectorizer.fit_transform([
            resume_text,
            job_text
        ])

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(float(similarity) * 100, 2)

    except ValueError:
        return 0.0


# --------------------------------------------------
# 5. RESUME ANALYSIS
# --------------------------------------------------

def analyze_resume(resume_text, job_text):

    resume_skills = normalize_skills(
        extract_skills(resume_text)
    )

    job_skills = normalize_skills(
        extract_skills(job_text)
    )

    matching_skills = sorted(
        set(resume_skills) & set(job_skills)
    )

    missing_skills = sorted(
        set(job_skills) - set(resume_skills)
    )

    if job_skills:
        skill_coverage = round(
            len(matching_skills) / len(job_skills) * 100,
            2
        )
    else:
        skill_coverage = 0.0

    text_similarity = calculate_similarity(
        resume_text,
        job_text
    )

    # An experimental indicator, not hiring probability.
    profile_alignment = round(
        (skill_coverage * 0.6) +
        (text_similarity * 0.4),
        2
    )

    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "skill_coverage": skill_coverage,
        "text_similarity": text_similarity,
        "profile_alignment": profile_alignment
    }


# --------------------------------------------------
# 6. GROQ AI RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(
    resume_text,
    job_text,
    matching_skills,
    missing_skills
):

    if not GROQ_API_KEY:
        return (
            "Groq API key was not found. "
            "Please check your .env file."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
You are a professional resume and career assistant.

Analyze the following resume against the job description.

RESUME:
{resume_text[:10000]}

JOB DESCRIPTION:
{job_text[:6000]}

MATCHING SKILLS:
{", ".join(matching_skills) or "None detected"}

MISSING SKILLS:
{", ".join(missing_skills) or "None detected"}

Provide the response using these headings:

1. Resume Strengths
Identify strengths supported by the resume.

2. Areas to Improve
Identify genuine gaps compared with the job description.

3. Recommended Skills to Learn
Suggest the most relevant missing skills.
Do not say the candidate already knows them.

4. Resume Improvement Suggestions
Suggest specific improvements to wording, projects,
and presentation. Do not invent experience or achievements.

5. Project Suggestions
Suggest two realistic projects that could help demonstrate
the missing skills. Clearly label these as future project ideas.

Important rules:
- Never invent experience, projects, certifications or achievements.
- Never assume a missing skill is definitely unknown.
- Do not calculate or claim a probability of selection.
- Use simple, professional language.
- Keep recommendations relevant to the target job.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful resume analysis assistant. "
                        "Only make claims supported by the provided text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"Unable to generate AI recommendations: {error}"


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎯 AI RESUME")
    st.markdown("### INTELLIGENCE PLATFORM")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["Home", "Resume Analyzer", "About"]
    )

    st.markdown("---")
    st.caption("AI-powered resume analysis")
    st.caption("Built with Python and Streamlit")


# --------------------------------------------------
# 8. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">AI Resume Intelligence</div>
        <div class="hero-subtitle">
            Analyze your resume, identify skill gaps and improve
            your alignment with job requirements using AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📄 Resume Analysis</h3>
            <p>Extract and review skills from your PDF resume.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 Job Matching</h3>
            <p>Compare resume content with job requirements.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>🤖 AI Recommendations</h3>
            <p>Receive suggestions to improve your resume.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How it works")

    st.write("""
    1. Upload your resume in PDF format.
    2. Paste the job description.
    3. Analyze matching and missing skills.
    4. Review the AI-generated recommendations.
    """)

    if st.button("Start Resume Analysis"):
        st.info(
            "Choose Resume Analyzer from the navigation menu."
        )


# --------------------------------------------------
# 9. RESUME ANALYZER PAGE
# --------------------------------------------------

elif page == "Resume Analyzer":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Resume Analyzer</div>
        <div class="hero-subtitle">
            Compare your resume with your target job description.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### 1. Upload Resume")

        uploaded_file = st.file_uploader(
            "Upload your resume in PDF format",
            type=["pdf"]
        )

    with right:
        st.markdown("### 2. Enter Job Description")

        job_description = st.text_area(
            "Paste the job requirements here",
            height=220,
            placeholder=(
                "Paste the complete job description, "
                "including required skills and responsibilities..."
            )
        )

    analyze_button = st.button(
        "🔍 Analyze My Resume",
        use_container_width=True
    )

    if analyze_button:

        if uploaded_file is None:
            st.error("Please upload your resume PDF.")

        elif not job_description.strip():
            st.error("Please enter a job description.")

        else:

            with st.spinner("Analyzing your resume..."):

                try:
                    resume_text = extract_resume_text(
                        uploaded_file
                    )

                    if not resume_text.strip():
                        st.error(
                            "No readable text was found in this PDF. "
                            "Please upload a text-based PDF."
                        )
                        st.stop()

                    results = analyze_resume(
                        resume_text,
                        job_description
                    )

                    st.session_state["analysis_results"] = results
                    st.session_state["resume_text"] = resume_text
                    st.session_state["job_description"] = job_description
                    st.session_state.pop("ai_recommendations", None)

                except Exception as error:
                    st.error(f"Resume processing failed: {error}")

    # Display analysis results
    if "analysis_results" in st.session_state:

        results = st.session_state["analysis_results"]

        st.markdown("---")
        st.markdown("## 3. Resume Analysis Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Text Similarity",
            f"{results['text_similarity']}%"
        )

        c2.metric(
            "Skill Coverage",
            f"{results['skill_coverage']}%"
        )

        c3.metric(
            "Profile Alignment Indicator",
            f"{results['profile_alignment']}%"
        )

        st.caption(
            "These are text and skill comparison measurements. "
            "The Profile Alignment Indicator is an experimental "
            "weighted score, not a prediction of hiring or selection."
        )

        st.markdown("### 4. Skill Analysis")

        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:
            st.markdown("#### ✅ Matching Skills")

            if results["matching_skills"]:
                html = "".join(
                    f'<span class="skill-tag">{skill}</span>'
                    for skill in results["matching_skills"]
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No matching skills were detected.")

        with skill_col2:
            st.markdown("#### ⚠️ Missing Skills")

            if results["missing_skills"]:
                html = "".join(
                    f'<span class="missing-tag">{skill}</span>'
                    for skill in results["missing_skills"]
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.success("All detected job skills matched.")

        st.markdown("### 5. Skill Coverage Chart")

        chart_data = {
            "Category": ["Matching Skills", "Missing Skills"],
            "Count": [
                len(results["matching_skills"]),
                len(results["missing_skills"])
            ]
        }

        fig = px.pie(
            chart_data,
            names="Category",
            values="Count",
            hole=0.55,
            title="Job Description Skill Coverage"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 6. Skills Detected in Your Resume")

        if results["resume_skills"]:
            st.write(", ".join(results["resume_skills"]))
        else:
            st.info("No listed skills were detected in your resume.")

        st.markdown("---")
        st.markdown("## 7. AI Resume Recommendations")

        if st.button("🤖 Generate AI Recommendations"):

            with st.spinner(
                "Generating personalized recommendations..."
            ):

                recommendations = generate_recommendations(
                    st.session_state["resume_text"],
                    st.session_state["job_description"],
                    results["matching_skills"],
                    results["missing_skills"]
                )

                st.session_state["ai_recommendations"] = recommendations

        if "ai_recommendations" in st.session_state:
            st.markdown(
                st.session_state["ai_recommendations"]
            )


# --------------------------------------------------
# 10. ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">About the Project</div>
        <div class="hero-subtitle">
            An AI-assisted platform for resume and job description analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Overview")

    st.write("""
    AI Resume Intelligence helps candidates understand how their
    resumes align with job descriptions. It extracts resume text,
    identifies skills, compares job requirements and generates
    improvement recommendations.
    """)

    st.markdown("### Technologies Used")

    technologies = [
        ("Python", "Main programming language"),
        ("Streamlit", "Web application interface"),
        ("PyPDF", "PDF text extraction"),
        ("Regular Expressions", "Rule-based skill extraction"),
        ("Scikit-learn TF-IDF", "Text vectorization"),
        ("Cosine Similarity", "Text similarity measurement"),
        ("Groq API - GPT-OSS 20B", "AI-generated recommendations"),
        ("Plotly", "Interactive charts")
    ]

    for technology, purpose in technologies:
        st.markdown(
            f"**{technology}** — {purpose}"
        )

    st.markdown("---")

    st.markdown("### Important Accuracy Information")

    st.write("""
    This application provides automated text and skill comparisons.
    Its results depend on the wording of the resume, job description
    and the skills included in the skill dictionary.

    It is an assistance tool, not a recruitment decision system.
    It does not predict whether a candidate will be selected.
    """)
