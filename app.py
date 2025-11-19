import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and transformer
# -----------------------------
@st.cache_data
def load_model_and_transformer():
    model = joblib.load("hotel_lr_model.pkl")
    ct = joblib.load("column_transformer.pkl")
    return model, ct

model, ct = load_model_and_transformer()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("hotel_bookings_with_id.csv")  # Dataset must have 'customer_name'

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="🏨", layout="centered")
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Enter a **Customer Name** to see booking details and probability of cancellation.")

customer_name_input = st.text_input("Customer Name")

if st.button("Predict Booking Probability"):
    if not customer_name_input:
        st.error("❌ Please enter the Customer Name.")
    else:
        # Filter dataset for the customer
        customer_rows = df[df['customer_name'].str.lower() == customer_name_input.lower()]

        if customer_rows.empty:
            st.error("❌ Customer not found. Please check the name.")
        else:
            st.success(f"✅ Found {len(customer_rows)} booking(s) for {customer_name_input.title()}")

            for _, customer_row in customer_rows.iterrows():
                # Drop target column
                X_customer = customer_row.drop(labels=['is_canceled'], errors='ignore')
                X_customer_df = pd.DataFrame([X_customer])

                # Transform features
                X_encoded = ct.transform(X_customer_df)

                # Predict probability
                prob = model.predict_proba(X_encoded)[0][1]  # probability of cancellation

                # Display booking details
                st.subheader("📄 Booking Details")
                st.dataframe(pd.DataFrame([customer_row])[[
                    'hotel', 'arrival_date_year', 'arrival_date_month',
                    'adults', 'children', 'babies', 'meal',
                    'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type',
                    'adr'
                ]])

                # Display probabilities
                st.subheader("📊 Booking Probability")
                st.progress(prob)
                st.write(f"💔 Cancellation probability: **{prob*100:.2f}%**")
                st.write(f"💚 Booking probability: **{100 - prob*100:.2f}%**")
