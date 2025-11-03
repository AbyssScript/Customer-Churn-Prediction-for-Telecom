Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn (leave the service) based on various features like contract type, payment method, tenure, and more.
It also includes a Streamlit web app for easy prediction and visualization.

Project Components

cust_churn_prediction.ipynb

    The complete Jupyter Notebook containing all data cleaning, exploratory data analysis (EDA), feature engineering, model training (using SMOTE and RandomizedSearchCV), and SHAP analysis.

app.py

    The Streamlit web application that serves the model. It provides a UI for:

    Single Customer Prediction: A detailed form to predict churn for one person and see actionable recommendations.

    Batch Prediction: A CSV upload feature with column mapping, which returns a downloadable CSV with predictions and recommended actions for every customer.

Model Artifacts

customer_churn_model.pkl: The trained RandomForestClassifier model.

encoders.pkl: The saved LabelEncoder objects used to process the categorical data.

How to Run

1.Clone this repository

git clone https://github.com/<your-username>/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

2.Install dependencies

pip install -r requirements.txt

3.Run the Streamlit app

streamlit run app.py

📊 Dataset

Dataset used: Telco Customer Churn from IBM sample datasets.
It includes features like gender, tenure, payment method, and service usage details.

🧰 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Streamlit

Matplotlib / SHAP (for model explainability)
