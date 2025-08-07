# 🧑‍💼 Candidate Recommendation Engine

A Streamlit-based intelligent tool for evaluating job applicants based on their resumes, using both basic content similarity and advanced AI-driven scoring.

---

## ✨ Key Features

- 📄 Upload resumes (PDF, DOCX, TXT or ZIP)
- 📝 Input job descriptions for matching
- 🧠 Content-based similarity using embeddings
- 🧮 Employability prediction using trained ML model
- 🤖 AI-powered candidate analysis via Gemini (Google AI)
- 📊 Interactive score comparison and insights
- 📥 Downloadable results for offline use

---

## 🧭 Two Approaches: Basic vs Advanced

### 🔹 **1. Basic Recommendation System**

- Uses **Sentence-BERT (SBERT)** embeddings for:
  - Encoding job descriptions and resumes
  - Computing cosine similarity between them
- Extracts skills using regex-based keyword patterns
- Provides quick and lightweight recommendations
- Great for fast evaluations without training models

### 🔸 **2. Advanced Recommendation System**

- Includes everything from the Basic system **plus**:
  - A **Random Forest-based employability prediction model** trained on structured candidate attributes
  - Keyword overlap and skill count calculations
  - Gemini AI-powered analysis using candidate text and extracted skills
  - A weighted scoring system combining multiple signals:
    - **50%** Resume–JD similarity
    - **30%** Predicted employability
    - **10%** Keyword match ratio
    - **10%** Normalized skill count

> ✅ Use the **Advanced mode** for detailed insights and robust hiring decisions.
> ⚡ Use the **Basic mode** for quick similarity scoring.

---

## How To Use Both Modes 🛠️

### For Basic Analysis:
1. Go to "Basic Recommendation" tab
2. Paste job description
3. Upload resumes (files/text/ZIP)
4. Get instant matching scores

### For Advanced Analysis:
1. **First-time setup**:
   - Upload `CRS_data.csv` (from GitHub repo)
   - Click "Train Model" in sidebar (takes 1-2 minutes)
   - Verify "Model is ready" status appears

2. **Regular use**:
   - Switch to "Advanced Recommendation" tab
   - Enter job requirements
   - Add candidate resumes
   - Click "Evaluate Candidates" for full analysis

## Model Training Details 🤖

### Why Random Forest?
- Handles mixed data types (categorical + numerical)
- Robust against overfitting
- Provides probability estimates for employability
- Feature importance analysis built-in

### Training Data Requirements:
```csv
Age,EdLevel,Gender,MainBranch,YearsCode,YearsCodePro,PreviousSalary,ComputerSkills,Employed
25-34,Bachelor,Male,Professional developer,5,3,75000,8,1
35-44,Master,Female,Not professional developer,12,0,0,5,0

## ⚙️ How to Use

### ▶️ Local Deployment

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/candidate-recommendation-engine.git
   cd candidate-recommendation-engine
