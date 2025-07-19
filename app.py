from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import numpy as np
import fitz  # PyMuPDF
import os
import re

app = Flask(__name__)

# Load sentence-transformer model
model = SentenceTransformer('all-mpnet-base-v2')
kw_model = KeyBERT(model)


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


# Extract resume sections
def extract_sections_from_text(text):
    text_lower = text.lower()

    sections = {
        "skills": "",
        "work_experience": "",
        "education": "",
        "certifications": "",
        "summary": ""
    }

    patterns = {
        "summary": r"(?si)professional summary\s*(.*?)(education|skill[s]? summary|skills|projects|$)",
        "education": r"(?si)education\s*(.*?)(project[s]?|internship[s]?|certification[s]?|achievement[s]?|$)",
        "skills": r"(?si)skill[s]?\s*(summary)?\s*(.*?)(project[s]?|internship[s]?|certification[s]?|education|$)",
        "work_experience": r"(?si)(experience|internship[s]?)\s*(.*?)(certification[s]?|project[s]?|education|$)",
        "certifications": r"(?si)certification[s]?\s*(.*?)(achievement[s]?|project[s]?|skills|education|$)"
    }

    for section, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            content = match.group(1 if section != "skills" else 2)
            sections[section] = content.strip()

    return sections


# Extract keywords using KeyBERT
def extract_keywords(text, top_n=20):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0].lower() for kw in keywords]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_input_option = request.form.get("job_input_option")
        job_text = request.form.get("job_description", "").strip()
        job_pdf = request.files.get("job_pdf")
        resume_file = request.files.get("resume")

        if not resume_file or not resume_file.filename.lower().endswith(".pdf"):
            return render_template("index.html", error="Please upload a valid resume PDF.")

        # Save resume
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        resume_path = os.path.join(upload_folder, resume_file.filename)
        resume_file.save(resume_path)
        resume_text = extract_text_from_pdf(resume_path)
        os.remove(resume_path)

        # Get job description
        job_description = None
        if job_input_option == "text" and job_text:
            job_description = job_text
        elif job_input_option == "pdf" and job_pdf and job_pdf.filename:
            job_pdf_path = os.path.join(upload_folder, job_pdf.filename)
            job_pdf.save(job_pdf_path)
            job_description = extract_text_from_pdf(job_pdf_path)
            os.remove(job_pdf_path)

        if not job_description:
            return render_template("index.html", error="Please provide a valid job description (Text or PDF).")

        # Extract and compare sections
        cv_sections = extract_sections_from_text(resume_text)
        section_labels = list(cv_sections.keys())
        section_texts = list(cv_sections.values())

        # Compute similarity
        job_embedding = model.encode(job_description, convert_to_tensor=True)
        cv_embedding = model.encode(section_texts, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(cv_embedding, job_embedding).squeeze(1).cpu().numpy()

        ranked = sorted(zip(section_labels, scores), key=lambda x: x[1], reverse=True)
        overall_score = np.mean(scores)

        # Suggested skill improvements
        resume_skills = extract_keywords(cv_sections.get("skills", ""), top_n=20)
        job_skills = extract_keywords(job_description, top_n=30)
        missing_skills = list(set(job_skills) - set(resume_skills))

        return render_template(
            "index.html",
            result=ranked,
            overall_score=overall_score,
            job_desc=job_description[:1000] + "..." if len(job_description) > 1000 else job_description,
            missing_skills=missing_skills
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
