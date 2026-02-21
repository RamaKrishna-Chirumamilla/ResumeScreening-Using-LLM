from flask import Flask, render_template, request, redirect
import os

from embedding import getemb
from resume_adder import readres
from similarity import cossim

app = Flask(__name__)

updir = "resumes"
os.makedirs(updir, exist_ok=True)

jd = """
Looking for a Python Developer with strong knowledge of
Python, Machine Learning, NLP, LLMs, TensorFlow, Transformers,
Data Structures, SQL, Git.
"""

jdemb = getemb(jd)

@app.route("/", methods=["GET", "POST"])
def index():
    res = []

    if request.method == "POST":

        if "addresumes" in request.form:
            fs = request.files.getlist("resumes")
            for f in fs:
                if f.filename.endswith(".pdf"):
                    f.save(os.path.join(updir, f.filename))
            return redirect("/")

        if "matchresumes" in request.form:
            for fn in os.listdir(updir):
                if fn.endswith(".pdf"):
                    p = os.path.join(updir, fn)

                    txt = readres(p)
                    remb = getemb(txt)

                    sc = cossim(jdemb, remb)
                    res.append((fn, round(float(sc), 2)))

            res.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=res)

if __name__ == "__main__":
    app.run(debug=True)