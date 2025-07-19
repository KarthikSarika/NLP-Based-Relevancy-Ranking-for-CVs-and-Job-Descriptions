# 📄 NLP-Based Relevancy Ranking for CVs and Job Descriptions

An AI-powered web application that uses Natural Language Processing (NLP) to evaluate how relevant a candidate's resume is to a given job description. It assists recruiters and job seekers by providing an automatic relevancy score and skill gap analysis based on semantic similarity and keyword presence.

---

## 🚀 Features

- Upload job description via text or PDF
- Upload resume (PDF format)
- Extracts and compares content using NLP models
- Generates:
  - Section-wise relevancy scores
  - Overall relevancy score
  - Suggested skills to learn (gap analysis)
- User-friendly web interface built with Flask
- Works completely offline once dependencies are installed

---

## 🧠 How It Works

1. **Resume & Job Description Parsing**
   - Extracts raw text from uploaded PDF files using `PyMuPDF`.
2. **Text Embedding**
   - Uses `SentenceTransformer` (`all-mpnet-base-v2`) to generate semantic embeddings.
3. **Relevancy Scoring**
   - Compares vectors using cosine similarity.
   - Provides section-wise and overall relevancy scores.
4. **Skill Suggestion**
   - Identifies missing or underrepresented keywords/skills by comparing the job description to the resume.

---

## 📂 Project Structure

```
├── app.py                  # Flask application logic
├── templates/
│   └── index.html          # Main HTML page
├── static/
│   └── style.css           # Styling for the app
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

---

## 💻 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KarthikSarika/NLP-Based-Relevancy-Ranking-for-CVs-and-Job-Descriptions.git
cd your-repo-name
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Then open your browser and go to:  
`http://127.0.0.1:5000/`

---

## 📦 Dependencies

- Flask
- sentence-transformers
- numpy
- PyMuPDF (`fitz`)
- scikit-learn (for keyword analysis)

You can install them using:
```bash
pip install flask sentence-transformers numpy PyMuPDF scikit-learn
```

---

## 🧪 Example Test

Use this sample job description to test:  
📁 `Cybersecurity Analyst Job Description` (you can paste the one provided earlier)

Upload any resume PDF for comparison.

---

## 📌 TODOs / Future Improvements

- Add support for DOCX file parsing
- Store candidate data in a database
- Rank multiple resumes for one job description
- Integration with job portals or ATS
- Add charts/visuals to show gaps more intuitively

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Anjaneya Karthik Naidu Sarika**  
Feel free to reach out or contribute to this project!
