# 🚀 Deployment Guide for OncoPredict AI

## ✅ Application Status

The OncoPredict AI application has been **successfully developed and tested locally**.

### ✨ Features Implemented

✅ Cancer risk prediction (0-100%)  
✅ Risk categorization (Low/Moderate/High)  
✅ Interactive risk gauge visualization  
✅ ROC curve analysis  
✅ Confusion matrix display  
✅ Feature importance analysis  
✅ SHAP explainability  
✅ PDF report generation  
✅ Medical disclaimer  
✅ Model comparison (Logistic Regression vs Random Forest)  
✅ Dark mode toggle  
✅ Professional medical dashboard UI

### 📊 Model Performance

- **Logistic Regression**: 93.86% accuracy
- **Random Forest**: 97.37% accuracy
- Both models trained and saved successfully

---

## 🌐 Deployment Options

### Option A: Streamlit Community Cloud (RECOMMENDED)

**Prerequisites:**
- GitHub account
- Streamlit Community Cloud account (free at https://streamlit.io/cloud)

**Steps:**

1. **Create GitHub Repository**
   - Go to https://github.com/new
   - Repository name: `oncopredict-ai`
   - Description: "🧬 Explainable Cancer Risk Prediction System with ML and SHAP"
   - Set to Public
   - Click "Create repository"

2. **Push Code to GitHub**
   ```bash
   cd /home/user/webapp
   git remote add origin https://github.com/YOUR_USERNAME/oncopredict-ai.git
   git branch -M main
   git push -u origin main
   ```
   
   **Note:** Replace `YOUR_USERNAME` with your actual GitHub username.
   
   If GitHub authentication is needed, you can:
   - Use the GitHub tab in the code sandbox interface
   - Generate a Personal Access Token at https://github.com/settings/tokens
   - Use: `git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/oncopredict-ai.git`

3. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/oncopredict-ai`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"
   
4. **Wait for Deployment**
   - Initial deployment takes 3-5 minutes
   - Models will be trained automatically on first run
   - You'll receive a public URL like: `https://YOUR-USERNAME-oncopredict-ai-app-xxxxx.streamlit.app`

---

### Option B: Render.com

**Steps:**

1. **Push to GitHub** (same as Option A, steps 1-2)

2. **Create Render Account**
   - Go to https://render.com
   - Sign up/Login with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your `oncopredict-ai` repository
   - Configure:
     - **Name**: `oncopredict-ai`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
     - **Instance Type**: Free
   - Click "Create Web Service"

4. **Wait for Deployment**
   - Takes 5-10 minutes
   - You'll receive a URL like: `https://oncopredict-ai.onrender.com`

---

### Option C: Railway.app

**Steps:**

1. **Push to GitHub** (same as Option A, steps 1-2)

2. **Deploy on Railway**
   - Go to https://railway.app
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose `oncopredict-ai`
   - Railway auto-detects Python and deploys
   - Add environment variable:
     - `PORT`: `8501`

3. **Configure Start Command**
   - Go to Settings → Deploy
   - Custom Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

4. **Generate Public URL**
   - Go to Settings → Networking
   - Click "Generate Domain"
   - You'll get a URL like: `https://oncopredict-ai.railway.app`

---

## 📋 Manual GitHub Setup (If Needed)

If you need to manually push to GitHub from the sandbox:

```bash
# Navigate to project
cd /home/user/webapp

# Configure git (replace with your details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/oncopredict-ai.git
git branch -M main

# Push with personal access token
# Generate token at: https://github.com/settings/tokens
git push -u origin main
```

---

## 🧪 Local Testing

The application is already running locally and can be tested at:

**Local URL**: http://localhost:8501

To restart the application:
```bash
cd /home/user/webapp
pm2 restart oncopredict-ai
pm2 logs oncopredict-ai --nostream
```

---

## 📝 Post-Deployment Checklist

After successful deployment:

✅ Visit the live URL  
✅ Test risk prediction with sample data  
✅ Verify all visualizations load correctly  
✅ Test PDF report generation  
✅ Check both model types (Logistic & Random Forest)  
✅ Verify SHAP explanations work  
✅ Test dark mode toggle  
✅ Review all tabs (Performance, Feature Importance, SHAP, Comparison, Report)  

---

## 🔧 Troubleshooting

### Models Not Loading
- Models are auto-generated on first run
- Wait 2-3 minutes for initial training
- Check logs for errors

### Memory Issues on Free Tier
- Streamlit Cloud free tier: 1GB RAM
- Models are small (~400KB total)
- Should work fine on free tier

### SHAP Explainability Errors
- SHAP requires more memory
- May timeout on very slow connections
- Retry if it fails once

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check Streamlit documentation: https://docs.streamlit.io
- Check deployment platform documentation

---

## 🎉 Congratulations!

Your OncoPredict AI application is ready for deployment!

**Next Steps:**
1. Choose a deployment platform (Streamlit Cloud recommended)
2. Follow the steps above to deploy
3. Share your live application URL

Good luck! 🚀
