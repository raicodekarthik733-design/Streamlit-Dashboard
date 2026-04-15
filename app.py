from pathlib import Path
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import sys


MODEL_PATH = Path("best_model (3).pkl")
FEATURES_PATH = Path("features.pkl")
DECISION_THRESHOLD = 0.84


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance between two latitude/longitude points in kilometers."""
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return float(6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


@st.cache_resource
def load_artifacts():
    """Load model and features with multiple loading strategies."""
    model = None
    feature_names = None
    
    # Strategy 1: Try joblib
    try:
        model = joblib.load(str(MODEL_PATH))
        st.success("Model loaded with joblib")
    except Exception as e:
        st.warning(f"Joblib load failed: {e}")
        # Strategy 2: Try pickle
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            st.success("Model loaded with pickle")
        except Exception as e2:
            st.error(f"Pickle load also failed: {e2}")
            return None, None
    
    # Load features
    try:
        feature_names = joblib.load(str(FEATURES_PATH))
        st.success("Features loaded with joblib")
    except Exception as e:
        st.warning(f"Joblib features load failed: {e}")
        try:
            with open(FEATURES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
            st.success("Features loaded with pickle")
        except Exception as e2:
            st.error(f"Pickle features load also failed: {e2}")
            return model, None
    
    return model, feature_names


def build_feature_row(
    amount: float,
    hour: int,
    day_of_week: int,
    month: int,
    age: int,
    category_enc: int,
    gender: str,
    city_pop: int,
    lat: float,
    long_value: float,
    merch_lat: float,
    merch_long: float,
    feature_names: list,
) -> pd.DataFrame:
    distance_km = haversine_distance_km(lat, long_value, merch_lat, merch_long)

    row = {
        "amt": float(amount),
        "amt_log": float(np.log1p(amount)),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "month": int(month),
        "is_weekend": int(day_of_week in [5, 6]),
        "is_night": int(hour >= 22 or hour <= 5),
        "age": int(age),
        "distance_km": float(distance_km),
        "category_enc": int(category_enc),
        "gender_enc": int(gender == "M"),
        "city_pop_log": float(np.log1p(city_pop)),
        "lat": float(lat),
        "long": float(long_value),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
    }

    return pd.DataFrame([[row[name] for name in feature_names]], columns=feature_names)


def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔎", layout="wide")
    st.title("Fraud Detection Dashboard")
    st.write("Use the inputs below to predict whether a transaction is likely fraudulent.")

    # Display system information for debugging
    with st.expander("System Information (for debugging)"):
        st.write(f"Python version: {sys.version}")
        st.write(f"Streamlit version: {st.__version__}")
        st.write(f"NumPy version: {np.__version__}")
        st.write(f"Pandas version: {pd.__version__}")
        st.write(f"Joblib version: {joblib.__version__}")
        st.write(f"Current directory: {Path.cwd()}")
        st.write(f"Model exists: {MODEL_PATH.exists()}")
        st.write(f"Features exist: {FEATURES_PATH.exists()}")
        if MODEL_PATH.exists():
            st.write(f"Model size: {MODEL_PATH.stat().st_size} bytes")
        if FEATURES_PATH.exists():
            st.write(f"Features size: {FEATURES_PATH.stat().st_size} bytes")

    # Check if files exist
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Please ensure 'best_model.pkl' is uploaded to your repository.")
        st.stop()
    
    if not FEATURES_PATH.exists():
        st.error(f"Features file not found: {FEATURES_PATH}")
        st.info("Please ensure 'features.pkl' is uploaded to your repository.")
        st.stop()

    # Load artifacts
    with st.spinner("Loading model and features..."):
        model, feature_names = load_artifacts()
    
    if model is None:
        st.error("Failed to load model. The file may be corrupted or incompatible.")
        st.info("Troubleshooting steps:\n1. Re-save the model using joblib.dump(model, 'best_model.pkl')\n2. Ensure scikit-learn versions match between training and deployment\n3. Try using pickle instead")
        st.stop()
    
    if feature_names is None:
        st.error("Failed to load features. The file may be corrupted or incompatible.")
        st.stop()

    with st.sidebar:
        st.header("Model Info")
        st.write(f"**Model file:** `{MODEL_PATH.name}`")
        st.write(f"**Feature file:** `{FEATURES_PATH.name}`")
        st.write(f"**Decision threshold:** `{DECISION_THRESHOLD}`")
        st.write(f"**Total features:** `{len(feature_names)}`")

    st.info(
        "Note: category_enc must be entered as the encoded numeric category value, "
        "because the original category label encoder was not saved with the model."
    )

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Transaction Details")
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.0, step=1.0)
            hour = st.selectbox("Hour of Day", options=list(range(24)), index=12)
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(day_names.keys()),
                index=4,
                format_func=lambda x: day_names[x],
            )
            month = st.selectbox(
                "Month",
                options=list(month_names.keys()),
                index=0,
                format_func=lambda x: month_names[x],
            )

        with col2:
            st.subheader("Customer Info")
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)
            category_enc = st.number_input("Category Encoded Value", min_value=0, value=0, step=1)
            gender = st.selectbox("Gender", options=["F", "M"])
            city_pop = st.number_input("City Population", min_value=1, value=50000, step=1000)

        with col3:
            st.subheader("Location Data")
            lat = st.number_input("Customer Latitude", value=40.7128, format="%.6f")
            long_value = st.number_input("Customer Longitude", value=-74.0060, format="%.6f")
            merch_lat = st.number_input("Merchant Latitude", value=40.7306, format="%.6f")
            merch_long = st.number_input("Merchant Longitude", value=-73.9352, format="%.6f")

        submitted = st.form_submit_button("Predict Fraud Risk", use_container_width=True)

    if submitted:
        try:
            input_df = build_feature_row(
                amount=amount,
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                age=age,
                category_enc=category_enc,
                gender=gender,
                city_pop=city_pop,
                lat=lat,
                long_value=long_value,
                merch_lat=merch_lat,
                merch_long=merch_long,
                feature_names=feature_names,
            )

            probability = float(model.predict_proba(input_df)[0][1])
            prediction = int(probability >= DECISION_THRESHOLD)

            st.markdown("---")
            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("FRAUDULENT TRANSACTION")
                    st.warning("This transaction shows high risk indicators.")
                else:
                    st.success("LEGITIMATE TRANSACTION")
                    st.info("This transaction appears to be legitimate.")

                st.metric("Fraud Probability", f"{probability:.2%}", 
                         delta=f"{(probability - DECISION_THRESHOLD)*100:.1f}% vs threshold")
                st.metric("Distance (km)", f"{input_df.loc[0, 'distance_km']:.2f}")

            with result_col2:
                st.subheader("Model Input Preview")
                st.dataframe(input_df.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

    with st.expander("Feature Names Used by the Model"):
        st.write(feature_names)


if __name__ == "__main__":
    main()
