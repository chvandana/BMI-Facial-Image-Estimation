# BMI Estimation from Facial Images

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-orange?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-brightgreen)](https://scikit-learn.org/)

---

## 📌 Overview
This project estimates **Body Mass Index (BMI)** from facial images using **machine learning** and **FaceNet** feature extraction.  
It predicts BMI directly from a user's facial image with real-time results.

---

## 🛠 Technologies Used
- **Python**  
- **Flask** (Web Application)  
- **FaceNet** (Facial Feature Extraction)  
- **Machine Learning Models**:
  - Random Forest Regressor  
  - Support Vector Regressor (SVR)  
  - Ridge Regression  
  - Ensemble Models  

---

## 📂 Project Structure

BMI-Estimation/
├── app.py # Flask web application
├── templates/ # HTML files
├── static/ # CSS and assets
├── models/ # Trained ML models (.pkl or .model)
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd BMI-Estimation

2. **Install dependencies**
pip install -r requirements.txt

3. Run the Flask app
python app.py

4. Open in browser
http://127.0.0.1:5000
