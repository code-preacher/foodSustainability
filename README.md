# 🌾 Sustainable Farming Practices Decision Support System (SFP DSS)
🌾 Sustainable Farming Practices Decision Support System (SFP DSS) — An AI‑driven web application for predicting wheat yield and recommending sustainable farming practices. Built with Flask, scikit‑learn, and a responsive HTML/CSS frontend, it integrates explainable AI (SHAP, LIME) and follows a full lifecycle from design to user evaluation.
---


## ✨ Features
- **Yield Prediction**: Regression model for continuous yield estimates.
- **Yield Classification**: Categorises yield into Low / Medium / High with probability scores.
- **Responsive Web UI**: Mobile‑friendly HTML/CSS frontend.
- **Real‑time Training & Prediction**: Train models on demand and get instant results.
- **Secure & Scalable**: Modular architecture for easy updates and dataset expansion.

---

## 🏗 System Architecture
**Three‑Tier Design**:
1. **Presentation Layer** – Web interface for data input and results display.
2. **Application Layer** – Preprocessing, model inference, and recommendation generation.
3. **Data Layer** – Secure storage of datasets, model files, and configuration.

---

## 🤖 Model Development
- **Data Prep**: Cleaning, encoding, SMOTE balancing, stratified splitting.
- **Baseline Models**: Decision Tree, Random Forest, Logistic Regression, SVM.
- **Final Model**: XGBoost + SVM soft voting ensemble.

---


## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Steps
```bash
# Clone the repository
git clone https://github.com/code-preacher/foodSustainability.git
cd foodSustainability

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
