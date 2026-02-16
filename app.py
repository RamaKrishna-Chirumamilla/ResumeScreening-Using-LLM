from flask import Flask, render_template, request, redirect
import os

from embedding import get_embedding
from resume_adder import read_resume
from similarity import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

job_description = """
Looking for a Python Developer with strong knowledge of
Python, Machine Learning, NLP, LLMs, TensorFlow, Transformers,
Data Structures, SQL, Git.
"""

jd_embedding = get_embedding(job_description)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":

        # Add resumes
        if "add_resumes" in request.form:
            files = request.files.getlist("resumes")
            for file in files:
                if file.filename.endswith(".pdf"):
                    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            return redirect("/")

        # Match resumes
        if "match_resumes" in request.form:
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.endswith(".pdf"):
                    path = os.path.join(UPLOAD_FOLDER, filename)

                    resume_text = read_resume(path)
                    resume_embedding = get_embedding(resume_text)

                    score = cosine_similarity(jd_embedding, resume_embedding)
                    results.append((filename, round(float(score), 2)))

            results.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
