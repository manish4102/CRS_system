# Candidate Recommendation System 🧑‍💼

## Dual-Analysis Approach 🔄

### 1. Basic Recommendation (Quick Analysis)
- **Purpose**: Fast preliminary screening
- **Features**:
  - Pure content-based matching
  - Skill-to-job description comparison
  - Gemini AI summary generation
- **Best for**: Quick candidate filtering before detailed evaluation

### 2. Advanced Recommendation (Trained Model)
- **Purpose**: Comprehensive evaluation
- **Features**:
  - Uses your **trained Random Forest model** (employability_model.pkl)
  - Hybrid scoring system:
    - 50% Content similarity
    - 30% Employability prediction
    - 20% Skill/keyword matching
  - Detailed AI analysis
- **Best for**: Final candidate selection with predictive insights

## How To Use Both Modes 🛠️

### For Basic Analysis:
1. Go to "Basic Recommendation" tab
2. Paste job description
3. Upload resumes (files/text/ZIP)
4. Get instant matching scores

### For Advanced Analysis:
1. **First-time setup**:
   - Upload `dataset.csv` (from GitHub repo)
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
