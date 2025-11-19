import streamlit as st
import pandas as pd
import joblib

# Load everything
@st.cache_resource
def load_all():
    model = joblib.load("hotel_lr_model.pkl")
    ct = joblib.load("column_transformer.pkl")
    df = pd.read_csv("hotel_bookings_demo.csv")
    return model, ct, df

model, ct, df = load_all()

# Beautiful UI
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="hotel", layout="centered")
st.title("Hotel Booking Cancellation Predictor")
st.success("Model & data loaded – Live and ready!")

st.markdown("### Search by customer name (1000 real bookings included)")
name = st.text_input("Enter any part of the name", placeholder="e.g. Smith, Garcia, Chen, Patel, John")

if st.button("Search & Predict", type="primary"):
    if not name.strip():
        st.error("Please type something")
    else:
        results = df[df['customer_name'].str.contains(name, case=False, na=False, regex=False)]
        if results.empty:
            st.warning("No match found. Try: Smith, Garcia, Johnson, Chen, Patel, Novak, Silva")
        else:
            st.balloons()
            st.write(f"**Found {len(results)} booking(s)!**")
            for _, row in results.iterrows():
                with st.expander(f"{row['customer_name']} – {row['hotel']} – {row['arrival_date_month']} {row['arrival_date_year']}"):
                    # Predict
                    X = row.drop('is_canceled', errors='ignore')
                    X_trans = ct.transform(pd.DataFrame([X]))
                    prob = model.predict_proba(X_trans)[0][1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Booking Details**")
                        st.write(f"Hotel: {row['hotel']}")
                        st.write(f"Arrival: {row['arrival_date_month']} {row['arrival_date_year']}")
                        st.write(f"Guests: {int(row['adults'])}+{int(row.get('children',0))}+{int(row.get('babies',0))}")
                        st.write(f"Price/night: ${row['adr']:.2f}")
                    with col2:
                        st.write("**Cancellation Risk**")
                        st.progress(prob)
                        st.metric("Probability", f"{prob*100:.1f}%")
                        if prob > 0.6:
                            st.error("HIGH RISK")
                        elif prob > 0.3:
                            st.warning("Moderate")
                        else:
                            st.success("Likely to arrive!")
