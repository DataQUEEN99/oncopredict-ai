# 🧬 OncoPredict AI - Explainable Cancer Risk Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced machine learning web application for cancer risk assessment with explainable AI capabilities, professional medical reporting, and comprehensive visualization tools.

## 🎯 Features

### Core Functionality
- **Risk Prediction**: Predicts cancer risk probability (0-100%) based on 9 key factors
- **Risk Categorization**: Classifies risk as Low (0-30%), Moderate (30-60%), or High (60-100%)
- **Visual Risk Gauge**: Interactive gauge chart showing risk level
- **Dual Model Support**: Compare Logistic Regression vs Random Forest predictions

### Advanced Analytics
- **ROC Curve Analysis**: Model performance evaluation with AUC score
- **Confusion Matrix**: Detailed classification performance metrics
- **Feature Importance**: Top 15 most influential factors visualization
- **SHAP Explainability**: Understand how each feature contributes to individual predictions
- **Model Comparison**: Side-by-side performance metrics and predictions

### User Experience
- **Professional Medical Dashboard**: Clean, healthcare-focused design
- **Dark Mode Toggle**: Comfortable viewing in any environment
- **PDF Report Generation**: Comprehensive downloadable assessment reports
- **Medical Disclaimer**: Clear ethical guidelines and limitations
- **Interactive Forms**: User-friendly input validation and guidance

## 📊 Input Features

### Demographics & Lifestyle
- Age (20-80 years)
- BMI (Body Mass Index)
- Smoking status
- Alcohol consumption
- Physical activity level (hours/week)

### Genetic & Clinical Markers
- Family history of cancer
- BRCA1 gene mutation
- TP53 gene mutation
- Tumor marker level (ng/mL)

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.31.0
- **ML Framework**: scikit-learn 1.4.0
- **Explainability**: SHAP 0.44.0
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Report Generation**: ReportLab
- **Data Processing**: Pandas, NumPy

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/oncopredict-ai.git
cd oncopredict-ai
```

**Note:** Replace `YOUR_USERNAME` with your GitHub username after pushing the code.

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
- Open your browser and navigate to: `http://localhost:8501`

## 📁 Project Structure

```
oncopredict-ai/
│
├── app.py                    # Main Streamlit application
├── model.py                  # ML model training and evaluation
├── data_processing.py        # Data loading and preprocessing
├── utils.py                  # Utility functions (visualization, PDF generation)
├── requirements.txt          # Python dependencies
│
├── models/                   # Trained models (auto-generated)
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── metrics.pkl
│
├── reports/                  # Generated PDF reports (auto-generated)
└── assets/                   # Static assets
```

## 📈 Model Performance

### Logistic Regression
- **Accuracy**: ~92%
- **Precision**: ~91%
- **Recall**: ~88%
- **ROC-AUC**: ~0.95

### Random Forest
- **Accuracy**: ~94%
- **Precision**: ~93%
- **Recall**: ~90%
- **ROC-AUC**: ~0.97

*Note: Actual performance may vary based on training data*

## 🔬 Data Source

This application uses the **UCI Breast Cancer Wisconsin Dataset** enhanced with synthetic lifestyle and genetic features for demonstration purposes.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This tool is for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- Risk scores are estimates based on machine learning models
- Always consult qualified healthcare professionals for medical decisions
- Do not ignore professional medical advice or delay seeking it
- Regular medical screenings are essential for cancer prevention
- If you have concerns about cancer risk, contact your healthcare provider immediately

## 🚀 Deployment

### Streamlit Community Cloud (Recommended)

1. **Fork/Push to GitHub**
2. **Visit**: https://streamlit.io/cloud
3. **Deploy**: Connect your repository and deploy
4. **Done**: Your app is live!

### Alternative Platforms
- **Render.com**: Full-stack deployment
- **Railway.app**: Simple container deployment
- **Heroku**: Classic PaaS deployment

## 📚 Usage Guide

1. **Enter Patient Information**
   - Fill in demographics and lifestyle factors
   - Provide genetic and clinical marker data

2. **Select Model**
   - Choose between Logistic Regression or Random Forest
   - Toggle dark mode if desired

3. **Analyze Risk**
   - Click "Analyze Cancer Risk" button
   - View risk score and category

4. **Explore Analysis**
   - Navigate through tabs for detailed insights
   - View model performance metrics
   - Understand feature importance
   - Explore SHAP explainability

5. **Generate Report**
   - Download comprehensive PDF report
   - Share with healthcare providers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Breast Cancer Wisconsin Dataset
- **Streamlit** for the amazing web framework
- **SHAP** for explainable AI capabilities
- **scikit-learn** for machine learning tools

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ by the OncoPredict AI Team**

*Empowering healthcare with explainable AI*
