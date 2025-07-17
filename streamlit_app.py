import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline

# Title of the app
st.title("Credit Card Default Prediction")

# Sidebar for input features
st.sidebar.title("Input Features")

# Basic Information Section
st.sidebar.subheader("Basic Information")
limit_bal = st.sidebar.number_input("Credit Limit (Limit_Balance)", min_value=10000, max_value=1000000, step=1000)
sex = st.sidebar.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
education = st.sidebar.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["Others", "Graduate", "University", "High School"][x])
marriage = st.sidebar.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: ["Married", "Single", "Others"][x-1])
age = st.sidebar.number_input("Age", min_value=21, max_value=100, step=1)

# Repayment Status Section
st.sidebar.subheader("Repayment Status")

# Define format function for repayment status
def format_repayment_status(x):
    status_map = {
        -2: "No consumption",
        -1: "Pay duly", 
        0: "Paid on-time",
        1: "Payment delay for 1 month",
        2: "Payment delay for 2 months",
        3: "Payment delay for 3 months",
        4: "Payment delay for 4 months",
        5: "Payment delay for 5 months",
        6: "Payment delay for 6 months",
        7: "Payment delay for 7 months",
        8: "Payment delay for 8 months",
        9: "Payment delay for 9+ months"
    }
    return status_map.get(x, f"Unknown ({x})")

pay_0 = st.sidebar.selectbox("Pay 0", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)
pay_2 = st.sidebar.selectbox("Pay 2", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)
pay_3 = st.sidebar.selectbox("Pay 3", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)
pay_4 = st.sidebar.selectbox("Pay 4", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)
pay_5 = st.sidebar.selectbox("Pay 5", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)
pay_6 = st.sidebar.selectbox("Pay 6", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], format_func=format_repayment_status)

# Bill Amounts Section
st.sidebar.subheader("Bill Amounts")
bill_amt1 = st.sidebar.number_input("Bill Amount 1", min_value=0, step=1000, max_value=1000000)
bill_amt2 = st.sidebar.number_input("Bill Amount 2", min_value=0, step=1000, max_value=1000000)
bill_amt3 = st.sidebar.number_input("Bill Amount 3", min_value=0, step=1000, max_value=1000000)
bill_amt4 = st.sidebar.number_input("Bill Amount 4", min_value=0, step=1000, max_value=1000000)
bill_amt5 = st.sidebar.number_input("Bill Amount 5", min_value=0, step=1000, max_value=1000000)
bill_amt6 = st.sidebar.number_input("Bill Amount 6", min_value=0, step=1000, max_value=1000000)

# Payment Amounts Section
st.sidebar.subheader("Payment Amounts")
pay_amt1 = st.sidebar.number_input("Pay Amount 1", min_value=0, step=1000, max_value=1000000)
pay_amt2 = st.sidebar.number_input("Pay Amount 2", min_value=0, step=1000, max_value=1000000)
pay_amt3 = st.sidebar.number_input("Pay Amount 3", min_value=0, step=1000, max_value=1000000)
pay_amt4 = st.sidebar.number_input("Pay Amount 4", min_value=0, step=1000, max_value=1000000)
pay_amt5 = st.sidebar.number_input("Pay Amount 5", min_value=0, step=1000, max_value=1000000)
pay_amt6 = st.sidebar.number_input("Pay Amount 6", min_value=0, step=1000, max_value=1000000)

# Create a DataFrame from user input
input_data = {
    "LIMIT_BAL": [limit_bal],
    "SEX": [sex],
    "EDUCATION": [education],
    "MARRIAGE": [marriage],
    "AGE": [age],
    "PAY_0": [pay_0],
    "PAY_2": [pay_2],
    "PAY_3": [pay_3],
    "PAY_4": [pay_4],
    "PAY_5": [pay_5],
    "PAY_6": [pay_6],
    "BILL_AMT1": [bill_amt1],
    "BILL_AMT2": [bill_amt2],
    "BILL_AMT3": [bill_amt3],
    "BILL_AMT4": [bill_amt4],
    "BILL_AMT5": [bill_amt5],
    "BILL_AMT6": [bill_amt6],
    "PAY_AMT1": [pay_amt1],
    "PAY_AMT2": [pay_amt2],
    "PAY_AMT3": [pay_amt3],
    "PAY_AMT4": [pay_amt4],
    "PAY_AMT5": [pay_amt5],
    "PAY_AMT6": [pay_amt6]
}


# Main content area for predictions
st.header("Prediction Results")

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame(input_data)

    # Initiate prediction
    prediction_pipeline = PredictionPipeline(input_df)
    probability, top_features_df = prediction_pipeline.run_pipeline()

    if probability[0][1] >= 0.5:
        # Light red background for default prediction
        st.markdown(
            f"""
            <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;">
                <h3 style="color: #d32f2f; margin: 0;">⚠️ Default Risk Detected</h3>
                <p style="color: #d32f2f; margin: 10px 0 0 0; font-size: 16px;">
                    The model predicts that the customer is likely to default with a probability of {probability[0][1] * 100:.2f}%.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Displaying top features influencing the prediction
        st.subheader("Top 5 Features Influencing the Prediction:")
        st.bar_chart(top_features_df.set_index('Feature')['Importance'] * 100, 
                     use_container_width=True, y_label="Importance (%)", x_label="Feature names")

    elif probability[0][0] > 0.5:
        # Light green background for no default prediction
        st.markdown(
            f"""
            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                <h3 style="color: #388e3c; margin: 0;">✅ Low Default Risk</h3>
                <p style="color: #388e3c; margin: 10px 0 0 0; font-size: 16px;">
                    The model predicts that the customer is not likely to default with a probability of {probability[0][0] * 100:.2f}%.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Displaying top features influencing the prediction
        st.subheader("Top 5 Features Influencing the Prediction:")
        st.bar_chart(top_features_df.set_index('Feature')['Importance'] * 100, 
                     use_container_width=True, y_label="Importance (%)", x_label="Feature names")

else:
    st.write("Please enter the input features and click 'Predict' to see the results.")


