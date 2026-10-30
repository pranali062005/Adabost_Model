import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Admission & Evaluation Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        padding: 1.5rem 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    'age', 'gender', 'state', 'family_income', 'high_school_gpa',
    'sat_score', 'act_score', 'attendance_rate', 'ap_courses',
    'extracurricular_count', 'volunteer_hours', 'leadership_positions',
    'coding_projects', 'social_media_hours', 'online_certifications',
    'essay_score', 'recommendation_score', 'interview_score'
]

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------------------------------
# NAVIGATION & SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Insights & Analytics"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")
model_file = st.sidebar.text_input("Pickle File Name", value="Adaboost_Model.pkl")

model = load_model(model_file)

if model is None:
    st.sidebar.warning(f"⚠️ Model file `{model_file}` not found. Ensure it's in the same folder.")
else:
    st.sidebar.success("✅ Model loaded successfully!")

# -----------------------------------------------------------------------------
# PAGE 1: SINGLE PREDICTION
# -----------------------------------------------------------------------------
if page == "Single Prediction":
    st.title("🎓 Student Evaluation & Prediction Portal")
    st.markdown("Fill in the candidate metrics below to evaluate admission success probability using the AdaBoost model.")

    if model is None:
        st.error(f"Please place your trained model file (`{model_file}`) in the working directory.")
    else:
        with st.form("student_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("👤 Demographics")
                age = st.number_input("Age", min_value=14, max_value=60, value=18)
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
                state = st.selectbox("State Code", options=["CA", "NY", "TX", "FL", "IL", "Other"])
                family_income = st.number_input("Family Income ($)", min_value=0, max_value=1000000, value=65000, step=5000)

                st.subheader("📚 Academic Scores")
                high_school_gpa = st.slider("High School GPA", 0.0, 4.0, 3.5, 0.01)
                sat_score = st.slider("SAT Score", 400, 1600, 1200, 10)
                act_score = st.slider("ACT Score", 1, 36, 26, 1)

            with col2:
                st.subheader("🏫 School & Coursework")
                attendance_rate = st.slider("Attendance Rate (%)", 0.0, 100.0, 92.0, 0.5)
                ap_courses = st.number_input("AP Courses Taken", min_value=0, max_value=20, value=3)
                online_certifications = st.number_input("Online Certifications", min_value=0, max_value=20, value=2)

                st.subheader("⚡ Extracurriculars & Skills")
                extracurricular_count = st.number_input("Extracurricular Activities", 0, 15, 3)
                volunteer_hours = st.number_input("Volunteer Hours", 0, 1000, 50)
                leadership_positions = st.number_input("Leadership Positions", 0, 10, 1)
                coding_projects = st.number_input("Coding Projects", 0, 20, 2)
                social_media_hours = st.slider("Daily Social Media Hours", 0.0, 12.0, 2.5, 0.5)

            with col3:
                st.subheader("📝 Assessments & Interviews")
                essay_score = st.slider("Essay Score (1-10)", 1.0, 10.0, 7.5, 0.1)
                recommendation_score = st.slider("Recommendation Score (1-10)", 1.0, 10.0, 8.0, 0.1)
                interview_score = st.slider("Interview Score (1-10)", 1.0, 10.0, 7.0, 0.1)

            submit_button = st.form_submit_button("Run Evaluation")

        if submit_button:
            gender_encoded = 1 if gender == "Female" else (0 if gender == "Male" else 2)
            state_encoded = hash(state) % 50

            input_data = pd.DataFrame([{
                'age': age,
                'gender': gender_encoded,
                'state': state_encoded,
                'family_income': family_income,
                'high_school_gpa': high_school_gpa,
                'sat_score': sat_score,
                'act_score': act_score,
                'attendance_rate': attendance_rate,
                'ap_courses': ap_courses,
                'extracurricular_count': extracurricular_count,
                'volunteer_hours': volunteer_hours,
                'leadership_positions': leadership_positions,
                'coding_projects': coding_projects,
                'social_media_hours': social_media_hours,
                'online_certifications': online_certifications,
                'essay_score': essay_score,
                'recommendation_score': recommendation_score,
                'interview_score': interview_score
            }])[FEATURE_NAMES]

            # Model Prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.header("📊 Evaluation Results")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="Predicted Class / Status",
                    value=f"Class {prediction}"
                )
                if probabilities is not None:
                    confidence = probabilities[prediction] * 100
                    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

            with res_col2:
                if probabilities is not None:
                    prob_df = pd.DataFrame({
                        'Probability': probabilities
                    }, index=[f"Class {i}" for i in range(len(probabilities))])
                    st.subheader("Class Probability Distribution")
                    st.bar_chart(prob_df)

# -----------------------------------------------------------------------------
# PAGE 2: BATCH PREDICTION
# -----------------------------------------------------------------------------
elif page == "Batch Prediction":
    st.title("📁 Batch Candidate Processing")
    st.markdown("Upload a CSV file containing candidate data to generate predictions in bulk.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(batch_df.head(10))

        missing_cols = [col for col in FEATURE_NAMES if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required feature columns in CSV: {missing_cols}")
        else:
            if st.button("Process Batch Predictions"):
                if model is not None:
                    preds = model.predict(batch_df[FEATURE_NAMES])
                    batch_df['Prediction'] = preds

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(batch_df[FEATURE_NAMES])
                        batch_df['Max_Probability'] = np.max(probs, axis=1)

                    st.success("Processing Complete!")
                    st.dataframe(batch_df)

                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_data,
                        file_name="candidate_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Model not loaded properly.")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL INSIGHTS & ANALYTICS
# -----------------------------------------------------------------------------
elif page == "Model Insights & Analytics":
    st.title("📈 Model Insights & Feature Importance")

    if model is None:
        st.error("Please ensure the model file is properly loaded.")
    else:
        st.subheader("Feature Importance Analysis")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                'Importance': importances
            }, index=FEATURE_NAMES).sort_values(by='Importance', ascending=True)

            st.bar_chart(fi_df)
        else:
            st.info("Feature importance data is not directly exposed by this estimator.")

        st.markdown("---")
        st.subheader("Model Metadata")
        st.json({
            "Estimator": str(type(model)),
            "Number of Estimators": getattr(model, "n_estimators", "N/A"),
            "Learning Rate": getattr(model, "learning_rate", "N/A"),
            "Number of Features Expected": getattr(model, "n_features_in_", len(FEATURE_NAMES))
        })