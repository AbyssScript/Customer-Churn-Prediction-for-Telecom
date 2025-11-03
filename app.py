import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL AND ARTIFACT LOADING ---
@st.cache_resource
def load_artifacts():
    """
    Loads the saved model, encoders, and feature names.
    This function is cached so it only runs once.
    """
    try:
        # Load the model and feature names (the 19 columns)
        with open("customer_churn_model.pkl", "rb") as f_model:
            model_data = pickle.load(f_model)
        model = model_data["model"]
        feature_names = model_data["features_names"] # This is X.columns

        # Load the encoders
        with open("encoders.pkl", "rb") as f_encoders:
            encoders = pickle.load(f_encoders)

        categorical_features = list(encoders.keys())
        numerical_features = [col for col in feature_names if col not in categorical_features]
        
        # Feature importances from your notebook's SHAP analysis
        # (Top 10 from the SHAP: Mean Feature Importance bar plot)
        feature_importances = {
            'Contract': 0.170,
            'OnlineSecurity': 0.105,
            'tenure': 0.095,
            'InternetService': 0.085,
            'TechSupport': 0.080,
            'MonthlyCharges': 0.075,
            'TotalCharges': 0.060,
            'OnlineBackup': 0.050,
            'PaymentMethod': 0.045,
            'Dependents': 0.040,
        }
        # Sort by importance for the chart (ascending for horizontal bar chart)
        importance_df = pd.DataFrame(
            feature_importances.items(), 
            columns=['Feature', 'Importance']
        ).sort_values(by='Importance', ascending=True)

        return model, encoders, feature_names, categorical_features, numerical_features, importance_df

    except FileNotFoundError as e:
        st.error(
            f"**ERROR: `{e.filename}` not found! 🔴**\n\n"
            "Please make sure you have run the `cust_churn_prediction.ipynb` notebook "
            "to generate `customer_churn_model.pkl` and `encoders.pkl` in the same directory as this app."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        st.stop()

# Load artifacts
model, encoders, feature_names, categorical_features, numerical_features, importance_df = load_artifacts()

# --- HELPER FUNCTION: KEY DRIVERS ---
def get_key_drivers(data_dict, return_lists=False):
    """
    Identifies key risk and retention factors based on the EDA/SHAP analysis.
    Can return detailed lists for UI or simple strings for batch processing.
    """
    risk_factors = []
    retention_factors = []

    # Risk Factors (based on notebook)
    if data_dict.get('Contract') == 'Month-to-month':
        risk_factors.append(("Month-to-Month Contract", "The #1 predictor of churn."))
    if data_dict.get('tenure', 0) < 12:
        risk_factors.append((f"Low Tenure ({data_dict.get('tenure', 0)} months)", "New customers are at high risk."))
    if data_dict.get('OnlineSecurity') == 'No':
        risk_factors.append(("No Online Security", "A strong churn indicator."))
    if data_dict.get('TechSupport') == 'No':
        risk_factors.append(("No Tech Support", "A strong churn indicator."))
    if data_dict.get('InternetService') == 'Fiber optic':
        risk_factors.append(("Fiber Optic Service", "Associated with higher churn than DSL."))
    if data_dict.get('PaymentMethod') == 'Electronic check':
        risk_factors.append(("Electronic Check Payment", "Associated with higher churn."))

    # Retention Factors (opposite of risks)
    if data_dict.get('Contract') == 'Two year':
        retention_factors.append(("Two Year Contract", "The strongest predictor of retention."))
    if data_dict.get('tenure', 0) > 48:
        retention_factors.append((f"High Tenure ({data_dict.get('tenure', 0)} months)", "Loyal customers rarely leave."))
    if data_dict.get('OnlineSecurity') == 'Yes':
        retention_factors.append(("Has Online Security", "A strong retention factor."))
    if data_dict.get('TechSupport') == 'Yes':
        retention_factors.append(("Has Tech Support", "A strong retention factor."))
    if data_dict.get('InternetService') == 'No':
        retention_factors.append(("No Internet Service", "These customers rarely churn."))

    if return_lists:
        return risk_factors, retention_factors
    
    # Return simple strings for batch processing
    top_risk = risk_factors[0][0] if risk_factors else "No Primary Risk"
    top_retention = retention_factors[0][0] if retention_factors else "No Primary Retention"
    return top_risk, top_retention

# --- NEW HELPER FUNCTION: BUSINESS RECOMMENDATIONS ---
def get_business_recommendations(data_dict, prediction_is_churn):
    """
    Generates actionable business recommendations based on notebook analysis.
    """
    # Only generate recommendations for high-risk (Churn) customers
    if not prediction_is_churn:
        return ["None (Low Risk Customer)"]
    
    recs = []
    
    # Use .get() for safety, providing a default value
    if data_dict.get('Contract') == 'Month-to-month':
        recs.append("Offer 1-year contract at a small discount.")
    
    if data_dict.get('tenure', 0) <= 6:
        recs.append("Enroll in new-customer onboarding/loyalty program.")
        
    if data_dict.get('OnlineSecurity') == 'No' or data_dict.get('TechSupport') == 'No':
        recs.append("Offer a 'Secure & Support' bundle discount.")
        
    if data_dict.get('InternetService') == 'Fiber optic':
        recs.append("Investigate service quality; offer proactive support.")
        
    if data_dict.get('PaymentMethod') == 'Electronic check':
        recs.append("Promote auto-pay via Credit Card for a $1/mo discount.")

    if data_dict.get('MonthlyCharges', 0) > 90: # Example, from "high monthly bills"
        recs.append("Review bill for loyalty discounts or plan optimization.")

    # If no specific rule triggered, but they are still a churn risk
    if not recs:
        recs.append("Standard retention offer (e.g., 1-month free Streaming TV).")
        
    return recs

# --- HELPER FUNCTION: COLUMN MAPPING GUESSER ---
def try_to_guess_index(col, options):
    """Helper to find the best match in the selectbox for column mapping. Case-insensitive."""
    if not options: # Handle empty list
        return 0
    col_lower = col.lower().replace("_", "").replace(" ", "")
    for i, option in enumerate(options):
        opt_lower = str(option).lower().replace("_", "").replace(" ", "")
        if opt_lower == col_lower:
            return i
    # Try a partial match if no exact match
    for i, option in enumerate(options):
        opt_lower = str(option).lower().replace("_", "").replace(" ", "")
        if col_lower in opt_lower:
            return i
    return 0 

# --- HELPER FUNCTION: DATA TO CSV (for download) ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- PAGE 1: SINGLE PREDICTION ---
def single_prediction_page():
    st.header("👤 Single Customer Prediction")
    st.markdown(
        "Fill in the customer's details below to get a churn prediction and detailed risk analysis."
    )

    with st.form("single_customer_form"):
        # Use columns and expanders to balance the layout
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**Customer Info**", expanded=True):
                gender = st.radio("Gender", ("Male", "Female"), horizontal=True)
                SeniorCitizen = st.checkbox("Senior Citizen")
                Partner = st.radio("Has Partner?", ("Yes", "No"), horizontal=True)
                Dependents = st.radio("Has Dependents?", ("Yes", "No"), horizontal=True)
            
            with st.expander("**Payment & Contract**", expanded=True):
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
                Contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
                PaperlessBilling = st.radio("Paperless Billing?", ("Yes", "No"), horizontal=True)
                PaymentMethod = st.selectbox(
                    "Payment Method",
                    ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"),
                )
                MonthlyCharges = st.number_input(
                    "Monthly Charges ($)", min_value=0.0, value=70.70, format="%.2f"
                )
                TotalCharges = st.number_input(
                    "Total Charges ($)", min_value=0.0, value=151.65, format="%.2f"
                )

        with col2:
            with st.expander("**Subscribed Services**", expanded=True):
                PhoneService = st.radio("Phone Service?", ("Yes", "No"), horizontal=True)
                MultipleLines = st.selectbox("Multiple Lines?", ("No", "Yes", "No phone service"))
                InternetService = st.selectbox("Internet Service?", ("DSL", "Fiber optic", "No"))
                OnlineSecurity = st.selectbox("Online Security?", ("No", "Yes", "No internet service"))
                OnlineBackup = st.selectbox("Online Backup?", ("No", "Yes", "No internet service"))
                DeviceProtection = st.selectbox("Device Protection?", ("No", "Yes", "No internet service"))
                TechSupport = st.selectbox("Tech Support?", ("No", "Yes", "No internet service"))
                StreamingTV = st.selectbox("Streaming TV?", ("No", "Yes", "No internet service"))
                StreamingMovies = st.selectbox("Streaming Movies?", ("No", "Yes", "No internet service"))

        # Submit Button
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    # --- Prediction Logic ---
    if submitted:
        input_data = {
            "gender": gender, "SeniorCitizen": 1 if SeniorCitizen else 0, "Partner": Partner,
            "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
            "MultipleLines": MultipleLines, "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
            "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
            "Contract": Contract, "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }

        input_df = pd.DataFrame([input_data])
        processed_df = input_df.copy()
        
        try:
            # Apply encoders
            for col in categorical_features:
                encoder = encoders[col]
                # Use .classes_ to check if the value is known before transforming
                if input_df[col].iloc[0] not in encoder.classes_:
                    st.error(f"Invalid value '{input_df[col].iloc[0]}' for feature '{col}'. "
                             "This value was not seen during model training.")
                    st.stop()
                processed_df[col] = encoder.transform(processed_df[col])
        except Exception as e:
            st.error(f"Error during encoding: {e}. Make sure the input values are valid.")
            st.stop()

        # Make Prediction
        prediction = model.predict(processed_df[feature_names])[0]
        probability = model.predict_proba(processed_df[feature_names])[0]
        churn_prob = probability[1] # Probability of 'Churn' (class 1)
        is_churn = (prediction == 1)

        # 5. Display Results
        st.write("---")
        st.header("📈 Prediction Result")
        
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            if is_churn:
                st.error("Prediction: **LIKELY TO CHURN**")
            else:
                st.success("Prediction: **LIKELY TO STAY**")
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                number={'suffix': '%', 'font': {'size': 40}},
                title={'text': "Churn Probability", 'font': {'size': 20}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF553B" if is_churn else "#00CC96"},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 204, 150, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(255, 255, 0, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(239, 85, 59, 0.2)'},
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=50, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        with result_col2:
            with st.container(border=True):
                st.subheader("Key Factors Analysis")
                risk_factors, retention_factors = get_key_drivers(input_data, return_lists=True)
                
                if is_churn:
                    st.markdown("This customer has several **high-risk factors**:")
                    if risk_factors:
                        for factor, reason in risk_factors:
                            st.markdown(f"- **{factor}**: *{reason}*")
                    else:
                        st.markdown("- *No primary risk factors identified, prediction may be based on a combination of minor factors.*")
                else:
                    st.markdown("This customer has several **strong retention factors**:")
                    if retention_factors:
                        for factor, reason in retention_factors:
                            st.markdown(f"- **{factor}**: *{reason}*")
                    else:
                        st.markdown("- *No primary retention factors identified, but no major risk factors are present.*")
            
            # --- NEW RECOMMENDATION BLOCK ---
            with st.container(border=True, height=150):
                st.subheader("Actionable Recommendations 🎯")
                recommendations = get_business_recommendations(input_data, is_churn)
                for rec in recommendations:
                    st.markdown(f"**- {rec}**")
            # --- END NEW ---


        # --- Detailed Analysis Expander ---
        st.write("")
        with st.expander("Show Detailed Model Analysis (from Notebook)"):
            st.markdown(
                """
                This analysis comes directly from the **SHAP feature importance** results 
                in the Jupyter notebook. It shows which features the model *learned*
                were the most powerful predictors of churn, on average.
                """
            )
            
            # Use the pre-calculated importance_df
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(color='#0068C9')
            ))
            fig.update_layout(
                title="Top 10 Most Important Features (SHAP Analysis)",
                xaxis_title="Average Impact on Churn Prediction (SHAP Value)",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                **Analysis:**
                * **`Contract`** is by far the most important feature. The model learned that "Month-to-month" contracts are the single biggest sign of a churn risk.
                * **`OnlineSecurity`**, **`tenure`**, **`InternetService`**, and **`TechSupport`** are the next most critical factors.
                * This shows that the model's prediction is heavily based on the customer's *contract terms* and their *core services/add-ons*, not just their billing amount.
                """
            )

# --- PAGE 2: BATCH PREDICTION ---
def batch_prediction_page():
    st.header("📄 Batch Customer Prediction")
    st.markdown("Upload a CSV file to predict churn for multiple customers at once.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="The CSV should contain columns similar to the Telco Churn dataset."
    )
    
    # Initialize session state keys
    if 'file_columns' not in st.session_state:
        st.session_state['file_columns'] = None
    if 'batch_df' not in st.session_state:
        st.session_state['batch_df'] = None
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['batch_df'] = df
            st.session_state['file_columns'] = df.columns.tolist()
            st.success(f"File '{uploaded_file.name}' uploaded successfully! ({len(df)} rows)")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state['file_columns'] = None # Clear state on error
            st.stop()

    # --- Step 2: Column Mapping (NOW INSIDE A CHECK) ---
    # This entire block will ONLY run if 'file_columns' exists in session state
    if st.session_state.get('file_columns'):
        st.write("---")
        st.subheader("Step 2: Map Your File's Columns")
        st.info(
            "Match the model's required features (left) to the columns from your uploaded file (right). "
            "We've tried to guess the correct mappings. \n\n"
            "**Note:** Any columns in your file not mapped (like 'customerID' or an existing 'Churn' column) will be **ignored by the model** "
            "but will be **kept in the final downloadable results file**."
        )

        file_cols = st.session_state['file_columns']
        
        with st.form("mapping_form"):
            mapping_dict = {}
            # Use 3 columns for a more compact mapping layout
            cols = st.columns(3)
            
            # Split features across 3 columns
            split_size = int(np.ceil(len(feature_names) / 3))
            for i, c in enumerate(cols):
                with c:
                    for col_name in feature_names[i*split_size:(i+1)*split_size]:
                        guess_index = try_to_guess_index(col_name, file_cols)
                        mapping_dict[col_name] = st.selectbox(
                            f"**{col_name}** (Model Feature)",
                            options=file_cols,
                            index=guess_index,
                            key=col_name
                        )
            
            map_submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

        # --- Step 3: Run Prediction & Show Results ---
        if map_submitted:
            with st.spinner("Processing data and running predictions..."):
                original_df = st.session_state['batch_df']
                
                try:
                    # 1. Create the dataframe with the model-expected column names
                    processed_df = pd.DataFrame()
                    for model_col, user_col in mapping_dict.items():
                        processed_df[model_col] = original_df[user_col]
                    
                    # 2. Store a copy of this for safe numerical conversion
                    num_data_for_conversion = processed_df[numerical_features].copy()
                    
                    # 3. --- NEW: Get Drivers & Recs from the ORIGINAL data ---
                    # We do this *before* encoding, so we have the string values
                    drivers = processed_df.apply(lambda row: get_key_drivers(row.to_dict()), axis=1)
                    
                    # 4. Now, apply encoders to the categorical data
                    for col in categorical_features:
                        encoder = encoders[col]
                        # Check for unseen values before transforming
                        unseen_values = set(processed_df[col]) - set(encoder.classes_)
                        if unseen_values:
                            st.error(f"Error in column '{col}': Found values {unseen_values} "
                                     "that were not present during model training. Please clean your data.")
                            st.stop()
                        processed_df[col] = encoder.transform(processed_df[col])
                    
                    # 5. Convert numerical data
                    for col in numerical_features:
                        processed_df[col] = pd.to_numeric(num_data_for_conversion[col], errors='coerce')
                    
                    if processed_df.isnull().values.any():
                        st.warning("Some non-numeric values were found and converted to 0.")
                        processed_df = processed_df.fillna(0)

                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.error("Please check your column mappings and data types.")
                    st.stop()
                
                # 6. Make Predictions
                predictions = model.predict(processed_df[feature_names])
                probabilities = model.predict_proba(processed_df[feature_names])[:, 1] # Churn prob

                # 7. Add all results to the original DataFrame
                results_df = original_df.copy()
                results_df['Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
                results_df['Churn_Probability'] = probabilities
                
                # 8. --- NEW: Add Drivers & Recs to the final dataframe ---
                results_df['Top_Risk_Factor'] = [d[0] for d in drivers]
                results_df['Top_Retention_Factor'] = [d[1] for d in drivers]
                
                # Get recommendations *based on the prediction*
                recs = []
                for i, row in processed_df.iterrows():
                    is_churn = (predictions[i] == 1)
                    # Need to get original string data for the recs function
                    original_row_data = row.to_dict()
                    # But the function needs strings, not encoded numbers. Re-map from original.
                    original_row_data = processed_df.iloc[i].to_dict()
                    rec_list = get_business_recommendations(original_row_data, is_churn)
                    recs.append(rec_list[0]) # Get the first/top recommendation
                
                results_df['Recommended_Action'] = recs
                
                st.session_state['results_df'] = results_df # Save for download

            # --- Display Results Dashboard ---
            st.write("---")
            st.subheader("Step 3: Prediction Results")
            
            total = len(results_df)
            churn_count = (predictions == 1).sum()
            no_churn_count = total - churn_count
            churn_rate = (churn_count / total) * 100

            st.metric(
                "Overall Churn Risk", 
                f"{churn_rate:.1f}%", 
                f"{churn_count} of {total} customers are predicted to churn.",
                delta_color="inverse"
            )

            # Pie Chart
            pie_data = pd.DataFrame({
                'Prediction': ['Churn', 'No Churn'],
                'Count': [churn_count, no_churn_count]
            })
            
            fig = go.Figure(data=[go.Pie(
                labels=pie_data['Prediction'], 
                values=pie_data['Count'],
                pull=[0.1 if c == 'Churn' else 0 for c in pie_data['Prediction']],
                marker_colors=['#EF553B', '#00CC96']
            )])
            fig.update_layout(
                title_text="Prediction Summary",
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download Button
            csv_data = convert_df_to_csv(results_df)
            st.download_button(
                label="📥 Download Full Results as CSV (with Recommendations)",
                data=csv_data,
                file_name=f"churn_predictions_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

            # Data Table
            st.dataframe(results_df, use_container_width=True)

# --- MAIN APP LOGIC ---
st.title("📊 Customer Churn Prediction App")
st.markdown("An interactive app to predict customer churn based on the model built in the notebook.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a Prediction Mode", 
    ["👤 Single Customer", "📄 Batch Upload (CSV)"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses the `customer_churn_model.pkl` and `encoders.pkl` "
    "files generated by your Jupyter Notebook."
)

# Page Routing
if page == "👤 Single Customer":
    # Clear batch state if we switch to single
    st.session_state['batch_df'] = None
    st.session_state['file_columns'] = None
    st.session_state['results_df'] = None
    single_prediction_page()
else:
    batch_prediction_page()

