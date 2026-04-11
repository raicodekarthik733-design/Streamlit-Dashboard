from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "features.pkl"
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
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
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
    feature_names: list[str],
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

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("Model file or features file is missing in this folder.")
        st.stop()

    model, feature_names = load_artifacts()

    with st.sidebar:
        st.header("Model Info")
        st.write(f"Model file: `{MODEL_PATH.name}`")
        st.write(f"Feature file: `{FEATURES_PATH.name}`")
        st.write(f"Decision threshold: `{DECISION_THRESHOLD}`")
        st.write(f"Total features used: `{len(feature_names)}`")

    st.info(
        "Note: `category_enc` must be entered as the encoded numeric category value, "
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
            amount = st.number_input("Transaction Amount", min_value=0.0, value=120.0, step=1.0)
            hour = st.selectbox("Hour", options=list(range(24)), index=12)
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
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)
            category_enc = st.number_input("Category Encoded Value", min_value=0, value=0, step=1)
            gender = st.selectbox("Gender", options=["F", "M"])
            city_pop = st.number_input("City Population", min_value=1, value=50000, step=1000)

        with col3:
            lat = st.number_input("Customer Latitude", value=40.7128, format="%.6f")
            long_value = st.number_input("Customer Longitude", value=-74.0060, format="%.6f")
            merch_lat = st.number_input("Merchant Latitude", value=40.7306, format="%.6f")
            merch_long = st.number_input("Merchant Longitude", value=-73.9352, format="%.6f")

        submitted = st.form_submit_button("Predict Fraud Risk")

    if submitted:
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

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Prediction: Fraudulent Transaction")
            else:
                st.success("Prediction: Legitimate Transaction")

            st.metric("Fraud Probability", f"{probability:.2%}")
            st.metric("Distance (km)", f"{input_df.loc[0, 'distance_km']:.2f}")

        with result_col2:
            st.subheader("Model Input Preview")
            st.dataframe(input_df, use_container_width=True)

    with st.expander("Feature Names Used by the Model"):
        st.write(feature_names)


if __name__ == "__main__":
    main()
