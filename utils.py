"""
Utility Functions for OncoPredict AI
Includes visualization, report generation, and helper functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import shap
import streamlit as st


def categorize_risk(probability):
    """
    Categorize cancer risk based on probability
    
    Args:
        probability: Risk probability (0-1)
    
    Returns:
        tuple: (category, color)
    """
    
    if probability < 0.3:
        return "Low Risk", "#27AE60"
    elif probability < 0.6:
        return "Moderate Risk", "#F39C12"
    else:
        return "High Risk", "#E74C3C"


def create_risk_gauge(probability):
    """
    Create a gauge chart showing risk level
    
    Args:
        probability: Risk probability (0-1)
    
    Returns:
        Plotly figure object
    """
    
    risk_category, risk_color = categorize_risk(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Cancer Risk Score<br><span style='font-size:0.8em'>{risk_category}</span>"},
        delta={'reference': 50, 'increasing': {'color': "#E74C3C"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#D5F4E6'},
                {'range': [30, 60], 'color': '#FCF3CF'},
                {'range': [60, 100], 'color': '#FADBD8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        font={'size': 16}
    )
    
    return fig


def plot_roc_curve(metrics, model_type='logistic'):
    """
    Plot ROC curve
    
    Args:
        metrics: Dictionary containing model metrics
        model_type: 'logistic' or 'random_forest'
    
    Returns:
        Matplotlib figure
    """
    
    roc_data = metrics[model_type]['roc_curve']
    auc_score = metrics[model_type]['roc_auc']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(roc_data['fpr'], roc_data['tpr'], 
            color='#2E86C1', linewidth=2, 
            label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(metrics, model_type='logistic'):
    """
    Plot confusion matrix
    
    Args:
        metrics: Dictionary containing model metrics
        model_type: 'logistic' or 'random_forest'
    
    Returns:
        Matplotlib figure
    """
    
    cm = metrics[model_type]['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=10):
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
    
    Returns:
        Matplotlib figure
    """
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_list = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors_list)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_shap_explanation(model, X, feature_names, model_type='logistic'):
    """
    Create SHAP explanation for the prediction
    
    Args:
        model: Trained model object
        X: Input features (single sample)
        feature_names: List of feature names
        model_type: 'logistic' or 'random_forest'
    
    Returns:
        SHAP values and figure
    """
    
    # Get the appropriate model
    trained_model = model.logistic_model if model_type == 'logistic' else model.rf_model
    
    # Create explainer
    explainer = shap.Explainer(trained_model, X)
    shap_values = explainer(X)
    
    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For binary classification, we want the SHAP values for the positive class (cancer)
    if len(shap_values.shape) == 3:
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values
    
    shap.plots.waterfall(shap_values_plot[0], show=False)
    
    plt.title('SHAP Waterfall Plot - Feature Contribution to Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return shap_values, fig


def generate_pdf_report(user_inputs, prediction_proba, risk_category, 
                        model_metrics, selected_model):
    """
    Generate PDF report of the cancer risk assessment
    
    Args:
        user_inputs: Dictionary of user input values
        prediction_proba: Predicted probability
        risk_category: Risk category string
        model_metrics: Model performance metrics
        selected_model: Selected model type
    
    Returns:
        BytesIO object containing PDF
    """
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86C1'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2E86C1'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🧬 OncoPredict AI", title_style))
    story.append(Paragraph("Cancer Risk Assessment Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Date and time
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment Summary
    story.append(Paragraph("Risk Assessment Summary", heading_style))
    
    risk_color = '#27AE60' if 'Low' in risk_category else '#F39C12' if 'Moderate' in risk_category else '#E74C3C'
    
    summary_data = [
        ['Risk Score:', f'{prediction_proba*100:.1f}%'],
        ['Risk Category:', risk_category],
        ['Model Used:', 'Logistic Regression' if selected_model == 'logistic' else 'Random Forest'],
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F6F7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    
    patient_data = [
        ['Age:', f"{user_inputs['age']} years"],
        ['BMI:', f"{user_inputs['bmi']:.1f}"],
        ['Smoking Status:', 'Yes' if user_inputs['smoking'] == 1 else 'No'],
        ['Alcohol Consumption:', 'Yes' if user_inputs['alcohol'] == 1 else 'No'],
        ['Physical Activity:', f"{user_inputs['physical_activity']} hours/week"],
        ['Family History:', 'Yes' if user_inputs['family_history'] == 1 else 'No'],
        ['BRCA1 Mutation:', 'Yes' if user_inputs['brca1_mutation'] == 1 else 'No'],
        ['TP53 Mutation:', 'Yes' if user_inputs['tp53_mutation'] == 1 else 'No'],
        ['Tumor Marker:', f"{user_inputs['tumor_marker']:.1f} ng/mL"],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F6F7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Model Performance
    story.append(Paragraph("Model Performance Metrics", heading_style))
    
    metrics = model_metrics[selected_model]
    
    performance_data = [
        ['Metric', 'Score'],
        ['Accuracy', f"{metrics['accuracy']:.3f}"],
        ['Precision', f"{metrics['precision']:.3f}"],
        ['Recall', f"{metrics['recall']:.3f}"],
        ['F1-Score', f"{metrics['f1']:.3f}"],
        ['ROC-AUC', f"{metrics['roc_auc']:.3f}"],
    ]
    
    performance_table = Table(performance_data, colWidths=[2.5*inch, 2.5*inch])
    performance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(performance_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Medical Disclaimer", heading_style))
    
    disclaimer_text = """
    <b>IMPORTANT MEDICAL DISCLAIMER:</b><br/><br/>
    
    This cancer risk assessment is generated by an artificial intelligence system for 
    educational and informational purposes only. It is NOT a substitute for professional 
    medical advice, diagnosis, or treatment.<br/><br/>
    
    <b>Key Points:</b><br/>
    • This tool uses machine learning models trained on historical data and should not be 
    used as the sole basis for medical decisions.<br/>
    • Risk scores are estimates and may not accurately reflect your actual cancer risk.<br/>
    • Always consult with qualified healthcare professionals for medical advice.<br/>
    • Do not ignore professional medical advice or delay seeking it because of information 
    from this tool.<br/>
    • Regular medical screenings and consultations are essential for cancer prevention and 
    early detection.<br/><br/>
    
    <b>If you have concerns about your cancer risk, please contact your healthcare provider 
    immediately.</b>
    """
    
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_text = f"<i>Generated by OncoPredict AI • {report_date}</i>"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


def format_metric_card(title, value, description=""):
    """
    Format a metric card for Streamlit display
    
    Args:
        title: Card title
        value: Main value to display
        description: Optional description
    
    Returns:
        HTML string for metric card
    """
    
    card_html = f"""
    <div style="
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    ">
        <h3 style="color: #2E86C1; margin: 0 0 10px 0;">{title}</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 0; color: #333;">{value}</p>
        {f'<p style="color: #666; margin: 10px 0 0 0; font-size: 14px;">{description}</p>' if description else ''}
    </div>
    """
    
    return card_html
