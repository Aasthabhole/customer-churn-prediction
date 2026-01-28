import streamlit as st
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

# -----------------------------
# Load Model Files
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction")
st.write("Fill customer details to predict whether the customer will **CHURN** or **STAY**.")

st.markdown("---")

# -----------------------------
# SQLite Setup
# -----------------------------
DB_FILE = "churn_history.db"

def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            senior_citizen TEXT,
            partner TEXT,
            dependents TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            phone_service TEXT,
            multiple_lines TEXT,
            internet_service TEXT,
            online_security TEXT,
            online_backup TEXT,
            device_protection TEXT,
            tech_support TEXT,
            streaming_tv TEXT,
            streaming_movies TEXT,
            contract TEXT,
            paperless_billing TEXT,
            payment_method TEXT,
            churn_probability REAL,
            prediction TEXT
        )
    """)

    conn.commit()
    conn.close()

create_table()

def insert_record(record: dict):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO customer_history (
            timestamp, gender, senior_citizen, partner, dependents, tenure,
            monthly_charges, total_charges, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, churn_probability, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["Timestamp"],
        record["Gender"],
        record["Senior Citizen"],
        record["Partner"],
        record["Dependents"],
        record["Tenure (months)"],
        record["Monthly Charges"],
        record["Total Charges"],
        record["Phone Service"],
        record["Multiple Lines"],
        record["Internet Service"],
        record["Online Security"],
        record["Online Backup"],
        record["Device Protection"],
        record["Tech Support"],
        record["Streaming TV"],
        record["Streaming Movies"],
        record["Contract"],
        record["Paperless Billing"],
        record["Payment Method"],
        record["Churn Probability (%)"],
        record["Prediction"]
    ))

    conn.commit()
    conn.close()

def fetch_history():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM customer_history ORDER BY id DESC", conn)
    conn.close()
    return df

def clear_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM customer_history")
    conn.commit()
    conn.close()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧾 Customer Details Form")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
    dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1, key="tenure")
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0, key="monthly")
    total_charges = monthly_charges * tenure
st.write(f"✅ Auto Calculated Total Charges: **{total_charges:.2f}**")


with col3:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_service")

    if phone_service == "No":
        multiple_lines = "No phone service"
        st.info("Multiple Lines: No phone service")
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], key="multiple_lines")

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")


def yes_no_internet_service(label, key):
    if internet_service == "No":
        return "No internet service"
    else:
        return st.selectbox(label, ["No", "Yes"], key=key)


st.markdown("---")
st.subheader("🌐 Internet Add-on Services")

col4, col5, col6 = st.columns(3)

with col4:
    online_security = yes_no_internet_service("Online Security", "online_security")
    online_backup = yes_no_internet_service("Online Backup", "online_backup")

with col5:
    device_protection = yes_no_internet_service("Device Protection", "device_protection")
    tech_support = yes_no_internet_service("Tech Support", "tech_support")

with col6:
    streaming_tv = yes_no_internet_service("Streaming TV", "streaming_tv")
    streaming_movies = yes_no_internet_service("Streaming Movies", "streaming_movies")


st.markdown("---")
st.subheader("🧾 Billing Details")

col7, col8 = st.columns(2)

with col7:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")

with col8:
    payment_method = st.selectbox(
        "Payment Method",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
        key="payment_method"
    )

# -----------------------------
# Convert Inputs → Model Features
# -----------------------------
input_data = {col: 0 for col in feature_names}

# Numeric columns
input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Dummies
input_data["gender_Male"] = 1 if gender == "Male" else 0
input_data["Partner_Yes"] = 1 if partner == "Yes" else 0
input_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0
input_data["PhoneService_Yes"] = 1 if phone_service == "Yes" else 0

# MultipleLines
if multiple_lines == "No phone service":
    input_data["MultipleLines_No phone service"] = 1
elif multiple_lines == "Yes":
    input_data["MultipleLines_Yes"] = 1

# InternetService
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data["InternetService_No"] = 1


def set_internet_feature(feature_yes, feature_no_internet, value):
    if value == "Yes":
        input_data[feature_yes] = 1
    elif value == "No internet service":
        input_data[feature_no_internet] = 1


set_internet_feature("OnlineSecurity_Yes", "OnlineSecurity_No internet service", online_security)
set_internet_feature("OnlineBackup_Yes", "OnlineBackup_No internet service", online_backup)
set_internet_feature("DeviceProtection_Yes", "DeviceProtection_No internet service", device_protection)
set_internet_feature("TechSupport_Yes", "TechSupport_No internet service", tech_support)
set_internet_feature("StreamingTV_Yes", "StreamingTV_No internet service", streaming_tv)
set_internet_feature("StreamingMovies_Yes", "StreamingMovies_No internet service", streaming_movies)

# Contract
if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# PaperlessBilling
input_data["PaperlessBilling_Yes"] = 1 if paperless == "Yes" else 0

# PaymentMethod
if payment_method == "Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"] = 1
elif payment_method == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_data["PaymentMethod_Mailed check"] = 1

# Create dataframe + scale
input_df = pd.DataFrame([input_data])[feature_names]
input_scaled = scaler.transform(input_df)

# -----------------------------
# Predict + Save to SQL
# -----------------------------
st.markdown("---")
st.subheader("🎯 Prediction Result")

if st.button("🔍 Predict Churn", key="predict_button"):

    prob = model.predict_proba(input_scaled)[0][1]
    prob_percent = prob * 100

    result = "CHURN" if prob >= 0.5 else "STAY"

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Churn Probability", f"{prob_percent:.2f}%")
    with m2:
        st.metric("Monthly Charges", f"{monthly_charges}")
    with m3:
        st.metric("Tenure", f"{tenure} months")

    st.write("📌 Churn Risk Level")
    st.progress(prob)

    if result == "CHURN":
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    # Readable input record
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure (months)": int(tenure),
        "Monthly Charges": float(monthly_charges),
        "Total Charges": float(total_charges),
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment_method,
        "Churn Probability (%)": round(prob_percent, 2),
        "Prediction": result
    }

    # Show readable table
    st.subheader("📝 Customer Inputs Used for Prediction")
    customer_inputs_df = pd.DataFrame(record.items(), columns=["Input", "Value"])
    st.dataframe(customer_inputs_df, use_container_width=True)

    # Save in SQLite
    insert_record(record)
    st.success("✅ Customer input saved into SQL database!")

# -----------------------------
# View Saved History from SQL
# -----------------------------
st.markdown("---")
st.subheader("📂 Saved Customer History (SQL Database)")

history_df = fetch_history()

if len(history_df) == 0:
    st.info("No saved history yet. Make a prediction to store the first customer.")
else:
    st.dataframe(history_df, use_container_width=True)

    st.download_button(
        "⬇️ Download History (CSV)",
        data=history_df.to_csv(index=False),
        file_name="customer_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear SQL History"):
        clear_history()
        st.warning("SQL History cleared! Refresh the page.")
