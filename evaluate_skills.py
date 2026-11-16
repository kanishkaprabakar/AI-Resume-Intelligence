
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from utils.skill_extractor import extract_skills

test_data = [
    {
        "text": "Experienced in Python, SQL, Pandas and NumPy.",
        "actual": ["python", "sql", "pandas", "numpy"]
    },
    {
        "text": "Knowledge of Java, machine learning and Git.",
        "actual": ["java", "machine learning", "git"]
    },
    {
        "text": "Worked with Python, Flask and MongoDB.",
        "actual": ["python", "flask", "mongodb"]
    }
]

all_skills = [
    "python", "sql", "pandas", "numpy",
    "java", "machine learning", "git",
    "flask", "mongodb", "tensorflow",
    "deep learning", "power bi"
]

y_true = []
y_pred = []

for item in test_data:

    predicted = extract_skills(item["text"])

    for skill in all_skills:
        y_true.append(int(skill in item["actual"]))
        y_pred.append(int(skill in predicted))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nSKILL EXTRACTION EVALUATION")
print("---------------------------")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print("\nTest resumes:", len(test_data))
print("Skills evaluated:", len(all_skills))
