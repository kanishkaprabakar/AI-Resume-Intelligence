# AI Resume Intelligence

An AI-powered resume analysis and job matching platform built using Python and Streamlit.

The application analyzes a candidate's resume against a target job description, identifies matching and missing skills, calculates text similarity, and generates AI-powered recommendations to improve the resume.

## 🚀 Features

- **Resume Upload:** Upload a resume in PDF format.
- **Skill Extraction:** Identify technical skills using keyword-based matching.
- **Job Description Analysis:** Extract required skills from a target job description.
- **Skill Gap Analysis:** Display matching skills and missing skills.
- **Text Similarity:** Compare resume content with the job description using TF-IDF and Cosine Similarity.
- **Profile Alignment Indicator:** Calculate an experimental score based on skill coverage and text similarity.
- **AI Recommendations:** Generate personalized resume improvement suggestions using Groq LLM.
- **Interactive Dashboard:** Display analysis results using Streamlit and Plotly.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Groq API | AI-powered recommendations |
| Plotly | Interactive visualizations |
| PyPDF | PDF text extraction |
| Regular Expressions | Rule-based skill extraction |
| Git & GitHub | Version control |

## ⚙️ How It Works

1. Upload a resume in PDF format.
2. Enter the target job description.
3. Extract resume text from the uploaded PDF.
4. Identify technical skills using keyword-based matching.
5. Compare resume skills with job requirements.
6. Calculate skill coverage and text similarity.
7. Generate improvement recommendations using an LLM.

## 📊 Analysis Methodology

**1. Skill Extraction**

Uses predefined skill keywords and regular expressions to identify technical skills in resumes and job descriptions.

**2. Text Similarity**

Uses TF-IDF vectorization to convert text into numerical representations and Cosine Similarity to measure textual similarity.

**3. Profile Alignment Indicator**

An experimental weighted score calculated using:

`Profile Alignment = (0.6 × Skill Coverage) + (0.4 × Text Similarity)`

This indicator helps compare resume alignment with a job description. It is not a hiring probability, trained ML prediction, or model accuracy.

**4. AI Recommendations**

Uses a pretrained language model through the Groq API to generate suggestions for improving resume alignment.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kanishkaprabakar/AI-Resume-Intelligence.git
```

### 2. Navigate to the project directory

```bash
cd AI-Resume-Intelligence
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root directory.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Get an API key from [Groq Console](https://console.groq.com/).

**Important:** Never upload your `.env` file or API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
AI-Resume-Intelligence/
│
├── app.py
├── evaluate_skills.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── utils/
    ├── pdf_reader.py
    └── skill_extractor.py
```

## 🔮 Future Enhancements

- Improve skill extraction using NLP techniques.
- Integrate semantic similarity using sentence embeddings.
- Support additional resume formats.
- Expand skill recognition across different job domains.
- Improve recommendation quality and evaluation.

