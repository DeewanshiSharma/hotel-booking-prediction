import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and transformer
# -----------------------------
@st.cache_resource
def load_model_and_transformer():
    model = joblib.load("hotel_lr_model.pkl")
    ct = joblib.load("column_transformer.pkl")
    return model, ct

model, ct = load_model_and_transformer()

# -----------------------------
# Load the DEMO dataset (the small one you uploaded)
# -----------------------------
try:
    df = pd.read_csv("hotel_bookings_demo.csv")   # ← CHANGED THIS LINE
    st.success("Model + demo data loaded successfully!")
except:
    df = pd.DataFrame()
    st.error("Demo file not found. Deploying... Refresh in 30 seconds.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="hotel", layout="centered")
st.title("Hotel Booking Cancellation Predictor")
st.markdown("### Enter a customer name to check their booking & cancellation risk")

customer_name_input = st.text_input("Customer Name (e.g. Smith, Garcia, John, Patel)")

if st.button("Check Booking & Predict Cancellation", type="primary"):
    if not customer_name_input.strip():
        st.error("Please enter a customer name.")
    else:
        if df.empty:
            st.warning("Data is still loading on Render. Please wait 30 seconds and try again.")
        else:
            customer_rows = df[df['customer_name'].str.contains(customer_name_input, case=False, na=False)]
            if customer_rows.empty:
                st.error("No booking found. Try names like: Smith, Garcia, Johnson, Chen, Patel, Silva")
            else:
                st.success(f"Found {len(customer_rows)} booking(s)!")
                st.balloons()
                for _, row in customer_rows.iterrows():
                    X = row.drop(labels=['is_canceled'], errors='ignore')
                    X_df = pd.DataFrame([X])
                    X_transformed = ct.transform(X_df)
                    prob = model.predict_proba(X_transformed)[0][1]

                    st.subheader("Booking Details")
                    details = row[['hotel', 'arrival_date_year', 'arrival_date_month', 'adults', 'children',
                                 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                                 'assigned_room_type', 'adr']].to_frame().T
                    st.dataframe(details)

                    st.subheader("Cancellation Probability")
                    st.progress(float(prob))
                    st.metric("Cancellation Risk", f"{prob*100:.1f}%")
                    if prob > 0.6:
                        st.error("High cancellation risk!")
                    elif prob > 0.3:
                        st.warning("Moderate risk")
                    else:
                        st.success("Low risk – likely to arrive!")
