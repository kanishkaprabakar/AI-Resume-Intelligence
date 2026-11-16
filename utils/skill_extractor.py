
import re

SKILLS_LIST = [
    # Programming languages
    "python", "java", "c++", "c", "javascript",

    # Databases
    "sql", "mysql", "mongodb", "postgresql",

    # Web development
    "html", "css", "react", "node.js",

    # AI and Machine Learning
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "predictive modeling",
    "scikit-learn",
    "tensorflow",
    "pytorch",

    # Generative AI
    "generative ai",
    "large language models",
    "llm",
    "llms",
    "prompt engineering",
    "rag",
    "retrieval augmented generation",
    "langchain",
    "langgraph",
    "ai agents",
    "vector database",
    "vector databases",
    "embeddings",
    "openai",
    "gemini",
    "claude",

    # Data libraries
    "pandas",
    "numpy",
    "matplotlib",

    # Frameworks and APIs
    "fastapi",
    "flask",
    "api",
    "rest api",

    # Tools
    "git",
    "github",
    "power bi",
    "excel",
    "docker",
    "aws",

    # Computer science fundamentals
    "data structures",
    "algorithms",
    "dsa",
    "oops",
    "dbms",
    "operating systems",
    "computer networks"
]


def extract_skills(text):

    text = text.lower()
    found_skills = []

    for skill in SKILLS_LIST:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills
