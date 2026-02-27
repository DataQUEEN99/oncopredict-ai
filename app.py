"""
🧬 OncoPredict AI - Explainable Cancer Risk Prediction System

A comprehensive machine learning web application for cancer risk assessment
with explainable AI capabilities and professional medical reporting.

Author: OncoPredict AI Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
from data_processing import load_and_prepare_data, create_input_features, get_feature_description
from model import CancerRiskModel, train_and_save_models
from utils import (
    categorize_risk, create_risk_gauge, plot_roc_curve, plot_confusion_matrix,
    plot_feature_importance, create_shap_explanation, generate_pdf_report
)

# Page configuration
st.set_page_config(
    page_title="OncoPredict AI - Cancer Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical dashboard design
def load_custom_css(dark_mode=False):
    """Load custom CSS styling"""
    
    bg_color = "#1E1E1E" if dark_mode else "#F4F6F7"
    text_color = "#FFFFFF" if dark_mode else "#2C3E50"
    card_bg = "#2C2C2C" if dark_mode else "#FFFFFF"
    
    st.markdown(f"""
    <style>
        .main {{
            background-color: {bg_color};
        }}
        
        .stApp {{
            background-color: {bg_color};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color};
            font-family: 'Helvetica Neue', sans-serif;
        }}
        
        .metric-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
        
        .risk-low {{
            color: #27AE60;
            font-weight: bold;
            font-size: 24px;
        }}
        
        .risk-moderate {{
            color: #F39C12;
            font-weight: bold;
            font-size: 24px;
        }}
        
        .risk-high {{
            color: #E74C3C;
            font-weight: bold;
            font-size: 24px;
        }}
        
        .info-box {{
            background-color: #D6EAF8;
            border-left: 5px solid #2E86C1;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .warning-box {{
            background-color: #FCF3CF;
            border-left: 5px solid #F39C12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .danger-box {{
            background-color: #FADBD8;
            border-left: 5px solid #E74C3C;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .stButton>button {{
            background-color: #2E86C1;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
            border: none;
            font-size: 16px;
        }}
        
        .stButton>button:hover {{
            background-color: #1F618D;
        }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_or_train_models():
    """Load pre-trained models or train new ones if not available"""
    
    models_dir = Path(__file__).parent / 'models'
    
    # Check if models exist
    if (models_dir / 'logistic_model.pkl').exists() and \
       (models_dir / 'scaler.pkl').exists():
        
        # Load existing models
        model = CancerRiskModel()
        model.load_models(str(models_dir))
        scaler = joblib.load(str(models_dir / 'scaler.pkl'))
        
        # Load feature names
        _, _, _, _, feature_names, _ = load_and_prepare_data()
        
        return model, scaler, feature_names
    
    else:
        # Train new models
        with st.spinner("Training models for the first time... This may take a minute."):
            model, scaler = train_and_save_models()
            _, _, _, _, feature_names, _ = load_and_prepare_data()
            
        return model, scaler, feature_names


def main():
    """Main application function"""
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
        st.title("🧬 OncoPredict AI")
        st.markdown("---")
        
        # Dark mode toggle
        dark_mode = st.checkbox("🌙 Dark Mode", value=False)
        
        # Model selection
        st.subheader("Model Selection")
        selected_model = st.radio(
            "Choose Model",
            options=['logistic', 'random_forest'],
            format_func=lambda x: 'Logistic Regression' if x == 'logistic' else 'Random Forest'
        )
        
        st.markdown("---")
        
        # Information
        st.info("**About OncoPredict AI**\n\nAn AI-powered cancer risk assessment tool using machine learning and explainable AI (SHAP) for transparent predictions.")
        
        # Medical disclaimer
        st.warning("⚠️ **Medical Disclaimer**\n\nThis tool is for educational purposes only. Always consult healthcare professionals for medical advice.")
    
    # Load custom CSS
    load_custom_css(dark_mode)
    
    # Load models
    try:
        model, scaler, feature_names = load_or_train_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Main header
    st.title("🧬 OncoPredict AI - Cancer Risk Prediction System")
    st.markdown("### Advanced Machine Learning for Explainable Cancer Risk Assessment")
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    # Left column - Input features
    with col1:
        st.header("📝 Patient Information")
        
        with st.form("patient_form"):
            st.subheader("Demographics & Lifestyle")
            
            age = st.slider("Age", min_value=20, max_value=80, value=45, step=1)
            bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=45.0, value=25.0, step=0.1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                alcohol = st.selectbox("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col_b:
                physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=10, value=3, step=1)
                family_history = st.selectbox("Family History of Cancer", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.subheader("Genetic & Clinical Markers")
            
            col_c, col_d = st.columns(2)
            with col_c:
                brca1_mutation = st.selectbox("BRCA1 Gene Mutation", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                tp53_mutation = st.selectbox("TP53 Gene Mutation", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col_d:
                tumor_marker = st.number_input("Tumor Marker Level (ng/mL)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
            
            st.markdown("---")
            
            # Submit button
            submit_button = st.form_submit_button("🔬 Analyze Cancer Risk", use_container_width=True)
    
    # Right column - Results
    with col2:
        st.header("📊 Risk Assessment Results")
        
        if submit_button:
            # Prepare user inputs
            user_inputs = {
                'age': age,
                'bmi': bmi,
                'smoking': smoking,
                'alcohol': alcohol,
                'physical_activity': physical_activity,
                'family_history': family_history,
                'brca1_mutation': brca1_mutation,
                'tp53_mutation': tp53_mutation,
                'tumor_marker': tumor_marker
            }
            
            # Create input features
            input_features = create_input_features(user_inputs, feature_names, scaler)
            
            # Make prediction
            with st.spinner("Analyzing risk factors..."):
                prediction_proba = model.predict(input_features, model_type=selected_model)
                risk_category, risk_color = categorize_risk(prediction_proba)
            
            # Display risk score
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
                <h2 style="color: white; margin: 0;">Risk Score: {prediction_proba*100:.1f}%</h2>
                <h3 style="color: white; margin: 10px 0 0 0;">{risk_category}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk gauge
            st.plotly_chart(create_risk_gauge(prediction_proba), use_container_width=True)
            
            # Risk interpretation
            if "Low" in risk_category:
                st.success("✅ **Low Risk Detected** - Continue regular health monitoring and maintain healthy lifestyle.")
            elif "Moderate" in risk_category:
                st.warning("⚠️ **Moderate Risk Detected** - Consider consulting with healthcare provider for preventive screenings.")
            else:
                st.error("🚨 **High Risk Detected** - Strongly recommend immediate consultation with oncology specialist.")
            
            # Store results in session state for other tabs
            st.session_state['prediction_made'] = True
            st.session_state['prediction_proba'] = prediction_proba
            st.session_state['risk_category'] = risk_category
            st.session_state['user_inputs'] = user_inputs
            st.session_state['input_features'] = input_features
            st.session_state['selected_model'] = selected_model
        
        else:
            st.info("👆 Please fill in patient information and click 'Analyze Cancer Risk' to see results.")
    
    # Tabs for additional analysis
    if 'prediction_made' in st.session_state and st.session_state['prediction_made']:
        st.markdown("---")
        st.header("🔍 Detailed Analysis")
        
        tabs = st.tabs([
            "📈 Model Performance", 
            "🎯 Feature Importance", 
            "🧠 SHAP Explainability",
            "🔄 Model Comparison",
            "📄 Generate Report"
        ])
        
        # Tab 1: Model Performance
        with tabs[0]:
            st.subheader("Model Performance Metrics")
            
            metrics = model.metrics[st.session_state['selected_model']]
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with metric_col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with metric_col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with metric_col4:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            
            # ROC Curve and Confusion Matrix
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(model.metrics, st.session_state['selected_model'])
                st.pyplot(fig_roc)
            
            with col_plot2:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(model.metrics, st.session_state['selected_model'])
                st.pyplot(fig_cm)
        
        # Tab 2: Feature Importance
        with tabs[1]:
            st.subheader("Feature Importance Analysis")
            
            importance_df = model.get_feature_importance(st.session_state['selected_model'])
            
            # Plot
            fig_importance = plot_feature_importance(importance_df, top_n=15)
            st.pyplot(fig_importance)
            
            # Table
            st.subheader("Feature Importance Scores")
            st.dataframe(importance_df.head(15), use_container_width=True)
        
        # Tab 3: SHAP Explainability
        with tabs[2]:
            st.subheader("SHAP Explainability Analysis")
            
            st.info("**What is SHAP?** SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction for this specific patient.")
            
            with st.spinner("Generating SHAP explanation..."):
                try:
                    # Convert input features to DataFrame for SHAP
                    input_df = pd.DataFrame(
                        st.session_state['input_features'], 
                        columns=feature_names
                    )
                    
                    shap_values, fig_shap = create_shap_explanation(
                        model, 
                        input_df,
                        feature_names,
                        st.session_state['selected_model']
                    )
                    
                    st.pyplot(fig_shap)
                    
                    st.markdown("""
                    **How to interpret:**
                    - Red bars push the prediction higher (increase cancer risk)
                    - Blue bars push the prediction lower (decrease cancer risk)
                    - Longer bars have stronger impact on the prediction
                    """)
                
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {str(e)}")
        
        # Tab 4: Model Comparison
        with tabs[3]:
            st.subheader("Model Comparison")
            
            # Compare metrics
            comparison_data = []
            for model_type in ['logistic', 'random_forest']:
                metrics = model.metrics[model_type]
                comparison_data.append({
                    'Model': 'Logistic Regression' if model_type == 'logistic' else 'Random Forest',
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Compare predictions for current input
            st.subheader("Prediction Comparison for Current Patient")
            
            pred_log = model.predict(st.session_state['input_features'], 'logistic')
            pred_rf = model.predict(st.session_state['input_features'], 'random_forest')
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.metric("Logistic Regression", f"{pred_log*100:.1f}%")
                cat_log, _ = categorize_risk(pred_log)
                st.write(f"**Category:** {cat_log}")
            
            with comp_col2:
                st.metric("Random Forest", f"{pred_rf*100:.1f}%")
                cat_rf, _ = categorize_risk(pred_rf)
                st.write(f"**Category:** {cat_rf}")
        
        # Tab 5: Generate Report
        with tabs[4]:
            st.subheader("📄 Download PDF Report")
            
            st.write("Generate a comprehensive PDF report of this cancer risk assessment.")
            
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = generate_pdf_report(
                            st.session_state['user_inputs'],
                            st.session_state['prediction_proba'],
                            st.session_state['risk_category'],
                            model.metrics,
                            st.session_state['selected_model']
                        )
                        
                        st.success("✅ PDF report generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"OncoPredict_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 20px;">
        <p><strong>🧬 OncoPredict AI</strong> - Explainable Cancer Risk Prediction System v1.0.0</p>
        <p><strong>⚠️ MEDICAL DISCLAIMER:</strong> This tool is for educational and informational purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.</p>
        <p>© 2024 OncoPredict AI Team. Built with Streamlit, scikit-learn, and SHAP.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
