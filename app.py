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
# Dataset not available on Render → show clean message
# -----------------------------
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="hotel", layout="centered")

st.title("Hotel Booking Cancellation Predictor")
st.success("Model loaded successfully and running live!")

st.markdown("""
### Search by Customer Name (Local Testing Only)
The full dataset (19 MB) is not included in this live deployment to keep it fast and free-tier compliant.

**On Render (this live version):** Name search is disabled  
**On your laptop (local):** Full search works when `hotel_bookings_with_id.csv` is present
""")

st.info("Live prediction form without dataset coming in next update — stay tuned!")
st.balloons()
