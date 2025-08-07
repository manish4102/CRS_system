import streamlit as st
import pandas as pd
import PyPDF2
import docx
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import re
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
import zipfile
import shutil

# Set page config at the very beginning (must be first Streamlit command)
st.set_page_config(
    page_title="Candidate Recommendation Engine", 
    layout="wide",
    page_icon="🧑‍💼"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Function to train and save the model
def train_and_save_model(dataset_path):
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Preprocessing
        categorical_cols = ['Age', 'EdLevel', 'Gender', 'MainBranch']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Feature selection
        features = ['Age', 'EdLevel', 'Gender', 'MainBranch', 
                   'YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
        target = 'Employed'
        
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        # Save artifacts
        joblib.dump(model, 'employability_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        return model, scaler, label_encoders, report
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None, None, None, None
    
def process_zip_file(zip_file):
    resumes = []
    temp_dir = None
    
    try:
        # Create a temporary directory
        temp_dir = os.path.join("/tmp", f"resumes_{hash(zip_file.name)}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the zip file to disk (needed for Streamlit Cloud)
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        # Process the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                # Skip macOS metadata and non-resume files
                if zip_info.filename.startswith('__MACOSX/') or zip_info.filename.startswith('._'):
                    continue
                
                if not zip_info.filename.lower().endswith(('.pdf', '.docx', '.txt')):
                    continue
                
                try:
                    # Extract and process each file
                    extracted_path = zip_ref.extract(zip_info, temp_dir)
                    with open(extracted_path, 'rb') as f:
                        text = extract_text_from_file(f)
                        if text:
                            resumes.append({
                                "filename": os.path.basename(zip_info.filename),
                                "text": text
                            })
                except Exception as e:
                    st.warning(f"Skipped {zip_info.filename}: {str(e)}")
                    continue
    
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    return resumes

# Improved text extraction with better PDF handling
def extract_text_from_file(file) -> Optional[str]:
    try:
        # Handle both uploaded files and file-like objects
        if hasattr(file, 'type'):  # Regular file upload
            file_type = file.type
            file_content = file.getvalue()
        else:  # File from ZIP or other source
            file_content = file.read() if hasattr(file, 'read') else file
            # Detect file type from extension
            filename = getattr(file, 'name', 'file').lower()
            if filename.endswith('.pdf'):
                file_type = 'application/pdf'
            elif filename.endswith(('.docx', '.doc')):
                file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif filename.endswith('.txt'):
                file_type = 'text/plain'
            else:
                st.error(f"Unsupported file format: {filename}")
                return None

        # Process based on file type
        if file_type == "application/pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([para.text for para in doc.paragraphs if para.text])
            
        elif file_type == "text/plain":
            return file_content.decode("utf-8") if isinstance(file_content, bytes) else file_content
            
    except Exception as e:
        filename = getattr(file, 'name', 'file')
        st.error(f"Error processing {filename}: {str(e)}")
        return None

# Enhanced skill extraction with better pattern matching
def extract_skills(text: str) -> List[str]:
    technical_skills = {
        'python': 'Python',
        'java': 'Java',
        'javascript|js': 'JavaScript',
        'typescript|ts': 'TypeScript',
        'c\+\+|cpp': 'C++',
        'c#|csharp': 'C#',
        'go|golang': 'Go',
        'rust': 'Rust',
        'sql': 'SQL',
        'nosql': 'NoSQL',
        'mongodb': 'MongoDB',
        'postgresql|postgres': 'PostgreSQL',
        'mysql': 'MySQL',
        'html': 'HTML',
        'css': 'CSS',
        'sass|scss': 'SASS',
        'react': 'React',
        'angular': 'Angular',
        'vue': 'Vue',
        'node\.?js|node': 'Node.js',
        'django': 'Django',
        'flask': 'Flask',
        'spring': 'Spring',
        'laravel': 'Laravel',
        'machine learning|ml': 'Machine Learning',
        'deep learning|dl': 'Deep Learning',
        'ai|artificial intelligence': 'AI',
        'tensorflow': 'TensorFlow',
        'pytorch': 'PyTorch',
        'data analysis': 'Data Analysis',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'spark': 'Spark',
        'hadoop': 'Hadoop',
        'cloud': 'Cloud',
        'aws': 'AWS',
        'azure': 'Azure',
        'gcp|google cloud': 'GCP',
        'docker': 'Docker',
        'kubernetes|k8s': 'Kubernetes',
        'devops': 'DevOps',
        'ci/cd': 'CI/CD',
        'jenkins': 'Jenkins',
        'git': 'Git',
        'terraform': 'Terraform'
    }
    
    found_skills = set()
    text_lower = text.lower()
    
    for pattern, skill_name in technical_skills.items():
        if re.search(rf'\b{pattern}\b', text_lower):
            found_skills.add(skill_name)
    
    return sorted(found_skills)

# Enhanced AI summary generation
def generate_ai_summary(job_desc: str, resume_text: str, skills: List[str]) -> str:
    if st.session_state.api_key is None:
        st.error("Gemini API key not configured")
        return None
    
    try:
        # Configure the Gemini client
        genai.configure(api_key=st.session_state.api_key)
        
        # Set up the model - USING GEMINI FLASH
        generation_config = {
            "temperature": 0.5,
            "max_output_tokens": 400,
        }
        
        prompt = f"""
        **Job Description:**
        {job_desc}
        
        **Candidate Skills:**
        {', '.join(skills)}
        
        **Resume Excerpt:**
        {resume_text[:2000]}
        
        Please provide a detailed 3-4 sentence analysis:
        1. How well the candidate's skills and experience match the job requirements
        2. Their strongest qualifications for this role
        3. Any potential gaps or areas that would need further evaluation
        4. Recommended next steps (interview focus areas, etc.)
        
        Be specific and reference actual skills/experiences mentioned.
        """
        
        # Using Gemini Flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        return response.text.strip()
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return None

# Improved employability prediction with better error handling
def predict_employability(resume_info: Dict[str, Any], model, scaler, label_encoders) -> float:
    try:
        # Create DataFrame with all required columns
        required_columns = ['Age', 'EdLevel', 'Gender', 'MainBranch', 
                          'YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
        
        # Fill missing values with defaults
        for col in required_columns:
            if col not in resume_info:
                resume_info[col] = 'unknown' if col in ['Age', 'EdLevel', 'Gender', 'MainBranch'] else 0
        
        df = pd.DataFrame([resume_info])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in df.columns:
                # Handle unseen labels
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else 'unknown')
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')
                df[col] = le.transform(df[col])
        
        # Scale features
        X = df[required_columns]
        X_scaled = scaler.transform(X)
        
        # Predict probability of being employed
        proba = model.predict_proba(X_scaled)[0][1]
        return round(proba * 100, 2)
    except Exception as e:
        st.error(f"Error predicting employability: {str(e)}")
        return 0.0

# Enhanced candidate evaluation with better scoring
def evaluate_candidates(job_desc: str, resumes: List[Dict[str, Any]], embedding_model, employability_model, scaler, label_encoders):
    candidates = []
    job_embedding = embedding_model.encode([job_desc])[0]
    job_keywords = set(re.findall(r'\b\w{4,}\b', job_desc.lower()))
    
    for i, resume in enumerate(resumes):
        text = resume['text']
        if not text:
            continue
            
        # Extract skills
        skills = extract_skills(text)
        skill_count = len(skills)
        
        # Calculate keyword matches
        resume_keywords = set(re.findall(r'\b\w{4,}\b', text.lower()))
        keyword_matches = len(job_keywords & resume_keywords)
        keyword_ratio = keyword_matches / len(job_keywords) if job_keywords else 0
        
        # Calculate experience estimates based on keyword matches
        experience_years = min(keyword_matches // 2, 20)  # Cap at 20 years
        pro_experience_years = min(keyword_matches // 3, 15)  # Cap at 15 years
        
        # Create resume info for prediction
        resume_info = {
            'Age': '25-35',  # Default age range
            'EdLevel': 'Bachelor',  # Default education level
            'Gender': 'unknown',
            'MainBranch': 'Professional developer' if pro_experience_years > 2 else 'Not professional developer',
            'YearsCode': experience_years,
            'YearsCodePro': pro_experience_years,
            'PreviousSalary': 0,
            'ComputerSkills': skill_count
        }
        
        # Calculate scores
        resume_embedding = embedding_model.encode([text])[0]
        similarity_score = cosine_similarity([job_embedding], [resume_embedding])[0][0] * 100
        employability_score = predict_employability(resume_info, employability_model, scaler, label_encoders)
        
        # Enhanced combined scoring
        combined_score = (
            0.5 * similarity_score +  # Content similarity
            0.3 * employability_score +  # Predicted employability
            0.1 * keyword_ratio * 100 +  # Keyword match ratio
            0.1 * (skill_count / 20 * 100)  # Normalized skill count (cap at 20)
        )
        
        candidates.append({
            "id": f"C-{i+1:03d}",
            "filename": resume['filename'],
            "text": text,
            "similarity_score": round(similarity_score, 1),
            "employability_score": round(employability_score, 1),
            "keyword_matches": keyword_matches,
            "skill_count": skill_count,
            "combined_score": round(combined_score, 1),
            "skills": skills,
            "experience_years": experience_years,
            "pro_experience_years": pro_experience_years
        })
    
    return sorted(candidates, key=lambda x: x["combined_score"], reverse=True)

# Main App
def main():
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Dataset upload for training
        st.subheader("Model Training")
        dataset_file = st.file_uploader("Upload training dataset (CSV)", type=['csv'])
        
        if dataset_file and st.button("Train Model"):
            with st.spinner("Training model..."):
                model, scaler, label_encoders, report = train_and_save_model(dataset_file)
                if model:
                    st.session_state.model_trained = True
                    st.session_state.models_loaded = True
                    st.success("Model trained successfully!")
                    st.text("Classification Report:")
                    st.text(report)
                    st.rerun()
        
        # API Key Setup
        st.subheader("OpenAI API Key")
        if st.session_state.api_key is None:
            api_key = st.text_input("Enter your OpenAI API key:", 
                                  type="password",
                                  key="api_key_input")
            if api_key:
                st.session_state.api_key = api_key
                st.success("API key configured")
                st.rerun()
        else:
            st.success("OpenAI API key is configured")
            if st.button("Change API Key"):
                st.session_state.api_key = None
                st.rerun()
        
        # Model information
        st.subheader("Model Status")
        if st.session_state.get('model_trained', False) or os.path.exists('employability_model.pkl'):
            st.success("Model is ready")
        else:
            st.warning("Model not trained yet")

    # Create tabs
    tab1, tab2 = st.tabs(["Advanced Recommendation", "Basic Recommendation"])
    
    with tab1:
        st.title("🧑‍💼 Advanced Candidate Recommendation")
        
        # Check if model is trained
        if not st.session_state.get('model_trained', False) and not os.path.exists('employability_model.pkl'):
            st.warning("Please train the model first using the sidebar")
            st.stop()
        
        # Load models
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            employability_model = joblib.load('employability_model.pkl')
            scaler = joblib.load('scaler.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()
        
        # Job description input
        with st.container():
            st.subheader("📝 Job Description")
            job_desc = st.text_area(
                "Paste the job description here:", 
                height=200,
                placeholder="Include required skills, qualifications, and responsibilities...",
                key="job_description_input"
            )
        
        # Resume upload options
        with st.container():
            st.subheader("📂 Upload Resumes")
            
            # Create three options for resume input
            upload_option = st.radio(
                "Choose how to provide resumes:",
                options=["Upload files", "Paste text", "Upload zip folder"],
                horizontal=True
            )
            
            resumes = []
            
            if upload_option == "Upload files":
                uploaded_files = st.file_uploader(
                    "Select resume files (PDF, DOCX, or TXT):",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'txt'],
                    key="resume_uploader"
                )
                if uploaded_files:
                    for file in uploaded_files:
                        text = extract_text_from_file(file)
                        if text:
                            resumes.append({
                                "filename": file.name,
                                "text": text
                            })
            
            elif upload_option == "Paste text":
                resume_text = st.text_area(
                    "Paste resume text here:",
                    height=200,
                    key="resume_text_input"
                )
                if resume_text.strip():
                    resumes.append({
                        "filename": "pasted_resume.txt",
                        "text": resume_text
                    })
            
            elif upload_option == "Upload zip folder":
                zip_file = st.file_uploader(
                    "Upload a zip file containing resumes:",
                    type=['zip'],
                    key="zip_uploader"
                )
                if zip_file:
                    with st.spinner("Extracting resumes from zip..."):
                        resumes = process_zip_file(zip_file)
                        st.success(f"Processed {len(resumes)} resumes from ZIP file")
                        try:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                temp_dir = "temp_resumes"
                                zip_ref.extractall(temp_dir)
                
                                for root, _, files in os.walk(temp_dir):
                                    for file in files:
                                        if not file.startswith('._'):  # Skip macOS metadata files
                                            file_path = os.path.join(root, file)
                                        if file.lower().endswith(('.pdf', '.docx', '.txt')):
                                            with open(file_path, 'rb') as f:
                                                text = extract_text_from_file(f)
                                                if text:
                                                    resumes.append({
                                                        "filename": file,
                                                        "text": text
                                                        })
                                shutil.rmtree(temp_dir)
                        except Exception as e:
                            st.error(f"Error processing zip file: {str(e)}")

        if st.button("🚀 Evaluate Candidates", type="primary") and job_desc and resumes:
            with st.spinner("Analyzing candidates..."):
                # Evaluate candidates
                candidates = evaluate_candidates(
                    job_desc, 
                    resumes, 
                    embedding_model, 
                    employability_model, 
                    scaler, 
                    label_encoders
                )
                
                # Display results with automatic AI summary
                st.subheader("🏆 Top Candidates")
                st.write(f"Found {len(candidates)} qualified candidates")
                
                # Score distribution
                st.write("### 📊 Score Comparison")
                score_df = pd.DataFrame({
                    'Candidate': [c['id'] for c in candidates],
                    'Content Match': [c['similarity_score'] for c in candidates],
                    'Employability': [c['employability_score'] for c in candidates],
                    'Keyword Matches': [c['keyword_matches'] for c in candidates],
                    'Skills Count': [c['skill_count'] for c in candidates],
                    'Combined Score': [c['combined_score'] for c in candidates]
                })
                st.bar_chart(score_df.set_index('Candidate'))
                
                # Display top candidates with automatic AI summary
                st.write("### 🔍 Candidate Details")
                for i, candidate in enumerate(candidates[:10]):
                    with st.expander(f"{i+1}. {candidate['id']} - {candidate['filename']} (Score: {candidate['combined_score']})"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Content Match", f"{candidate['similarity_score']}%")
                            st.metric("Employability", f"{candidate['employability_score']}%")
                            st.metric("Keyword Matches", candidate['keyword_matches'])
                            st.metric("Skills Count", candidate['skill_count'])
                            
                            st.subheader("Technical Skills")
                            st.write(", ".join(candidate['skills']))
                        
                        with col2:
                            # Automatic AI Analysis
                            if st.session_state.api_key:
                                with st.spinner("Generating AI analysis..."):
                                    analysis = generate_ai_summary(
                                        job_desc,
                                        candidate['text'],
                                        candidate['skills']
                                    )
                                    if analysis:
                                        st.subheader("AI Analysis")
                                        st.write(analysis)
                            
                            # Resume preview
                            st.subheader("Resume Preview")
                            st.text_area(
                                f"resume_preview_{i}",
                                value=candidate['text'][:1000] + ("..." if len(candidate['text']) > 1000 else ""),
                                height=300,
                                label_visibility="collapsed"
                            )
                
                # Download option
                st.download_button(
                    label="📥 Download Results",
                    data=score_df.to_csv(index=False),
                    file_name="candidate_scores.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.title("Basic Candidate Recommendation")
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Job description input
        job_desc = st.text_area("Job Description", height=200, 
                              placeholder="Paste the job description here...",
                              key="simple_job_desc")
        
        # Resume upload options
        st.subheader("Upload Candidate Resumes")
        
        # Create three options for resume input
        upload_option = st.radio(
            "Choose how to provide resumes:",
            options=["Upload files", "Paste text", "Upload zip folder"],
            horizontal=True,
            key="simple_upload_option"
        )
        
        resumes = []
        
        if upload_option == "Upload files":
            uploaded_files = st.file_uploader(
                "Choose files (PDF, DOCX, or TXT)", 
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt'],
                key="simple_resume_uploader"
            )
            if uploaded_files:
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    if text:
                        resumes.append({
                            "filename": file.name,
                            "text": text
                        })
        
        elif upload_option == "Paste text":
            resume_text = st.text_area(
                "Paste resume text here:",
                height=200,
                key="simple_resume_text_input"
            )
            if resume_text.strip():
                resumes.append({
                    "filename": "pasted_resume.txt",
                    "text": resume_text
                })
        
        elif upload_option == "Upload zip folder":
            zip_file = st.file_uploader(
                "Upload a zip file containing resumes:",
                type=['zip'],
                key="simple_zip_uploader"
            )
            if zip_file:
                with st.spinner("Extracting resumes from zip..."):
                    resumes = process_zip_file(zip_file)
                    st.success(f"Processed {len(resumes)} resumes from ZIP file")
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            temp_dir = "temp_resumes"
                            os.makedirs(temp_dir, exist_ok=True)
                            zip_ref.extractall(temp_dir)
                
                            for root, _, files in os.walk(temp_dir):
                                for filename in files:
                        # Skip macOS metadata files and non-resume files
                                    if filename.startswith('._') or not filename.lower().endswith(('.pdf', '.docx', '.txt')):
                                        continue
                            
                                    file_path = os.path.join(root, filename)
                                    try:
                                        with open(file_path, 'rb') as f:
                                            text = extract_text_from_file(f)
                                            if text:
                                                resumes.append({
                                                    "filename": filename,
                                                    "text": text
                                                })
                                    except Exception as e:
                                        st.error(f"Error processing file {filename}: {str(e)}")
                                        continue
                    except zipfile.BadZipFile:
                        st.error("Invalid ZIP file format")
                    except Exception as e:
                        st.error(f"Error processing ZIP file: {str(e)}")
                    finally:
            # Clean up temp directory
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)

        if st.button("Find Best Candidates", key="simple_find_candidates") and job_desc and resumes:
            with st.spinner("Processing candidates..."):
                # Process files
                candidates = []
                job_embedding = embedding_model.encode([job_desc])[0]
                
                for i, resume in enumerate(resumes):
                    text = resume['text']
                    # Calculate similarity
                    resume_embedding = embedding_model.encode([text])[0]
                    similarity_score = cosine_similarity([job_embedding], [resume_embedding])[0][0] * 100
                    
                    # Extract skills
                    skills = extract_skills(text)
                    
                    candidates.append({
                        "id": f"Candidate {i+1}",
                        "filename": resume['filename'],
                        "text": text,
                        "score": round(similarity_score, 1),
                        "skills": skills
                    })
                
                if not candidates:
                    st.error("No valid resumes found")
                    return
                
                # Sort by score
                candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
                
                # Display top candidates
                st.subheader("Top Candidates")
                
                for i, candidate in enumerate(candidates_sorted[:10]):
                    expander = st.expander(f"{i+1}. {candidate['id']} - {candidate['filename']} (Score: {candidate['score']})")
                    with expander:
                        st.write(f"**Similarity Score:** {candidate['score']}%")
                        st.write(f"**Skills:** {', '.join(candidate['skills'])}")
                        
                        # Automatic AI summary
                        if st.session_state.api_key:
                            with st.spinner("Generating AI analysis..."):
                                summary = generate_ai_summary(
                                    job_desc,
                                    candidate["text"],
                                    candidate["skills"]
                                )
                                if summary:
                                    st.write("**AI Summary:**")
                                    st.write(summary)
                        
                        # Show resume preview
                        st.write("**Resume Preview:**")
                        st.text_area(
                            f"simple_resume_preview_{i}",
                            value=candidate["text"][:500] + "...",
                            height=200,
                            label_visibility="collapsed"
                        )

if __name__ == "__main__":
    main()
