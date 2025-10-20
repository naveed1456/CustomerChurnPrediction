import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# --- CSS to emphasize Predict button ---
st.markdown("""
<style>
div.stButton > button, div.stForm form button {
    padding: 0.6rem 1.1rem;
    font-weight: 700;
    font-size: 1.05rem;
    border-radius: 10px;
}
div.stForm form button[type="submit"] {
    box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    transform: translateY(0);
}
div.stForm form button[type="submit"]:hover {
    box-shadow: 0 10px 18px rgba(0,0,0,0.18);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

FEATURE_ORDER = [
    "age", "gender", "tenure", "usage_frequency", "support_calls", "payment_delay",
    "total_spend", "last_interaction", "subscription_type_Basic",
    "subscription_type_Premium", "subscription_type_Standard",
    "contract_length_Annual", "contract_length_Monthly", "contract_length_Quarterly"
]

DEFAULTS = {
    'age': 20, 'gender': 0, 'tenure': 25, 'usage_frequency': 14, 'support_calls': 4,
    'payment_delay': 27, 'total_spend': 598, 'last_interaction': 9,
    'subscription_type_Basic': 1, 'subscription_type_Premium': 0, 'subscription_type_Standard': 0,
    'contract_length_Annual': 0, 'contract_length_Monthly': 1, 'contract_length_Quarterly': 0
}

@st.cache_resource(show_spinner=False)
def load_model(path="best_churn_model.joblib"):
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"Couldn't load model: {e}"

def make_form():
    st.subheader("Single Prediction")
    with st.form("single_form"):
        c1, c2 = st.columns(2)
        v = {}

        v['age'] = int(c1.number_input("Age", min_value=0, max_value=120,
                                       value=int(DEFAULTS['age']), key="k_age"))
        gender_choice = c2.selectbox("Gender", options=["Female", "Male"],
                                     index=int(DEFAULTS['gender']), key="k_gender_choice")
        v['gender'] = 1 if gender_choice == "Male" else 0

        v['tenure'] = int(c1.number_input("Tenure (months)", min_value=0,
                                          value=int(DEFAULTS['tenure']), key="k_tenure"))
        v['usage_frequency'] = int(c2.number_input("Usage Frequency", min_value=0,
                                                   value=int(DEFAULTS['usage_frequency']), key="k_usage_frequency"))
        v['support_calls'] = int(c1.number_input("Support Calls", min_value=0,
                                                 value=int(DEFAULTS['support_calls']), key="k_support_calls"))
        v['payment_delay'] = int(c2.number_input("Payment Delay (days)", min_value=0,
                                                 value=int(DEFAULTS['payment_delay']), key="k_payment_delay"))
        v['total_spend'] = float(c1.number_input("Total Spend", min_value=0.0,
                                                value=float(DEFAULTS['total_spend']), key="k_total_spend"))
        v['last_interaction'] = int(c2.number_input("Last Interaction (encoded)", min_value=0,
                                                    value=int(DEFAULTS['last_interaction']), key="k_last_interaction"))

        st.markdown("### Subscription Type")
        sub_choice = st.radio("Choose one", ["Basic", "Premium", "Standard"],
                              index=0, horizontal=True, key="k_sub_choice")
        v['subscription_type_Basic'] = 1 if sub_choice == "Basic" else 0
        v['subscription_type_Premium'] = 1 if sub_choice == "Premium" else 0
        v['subscription_type_Standard'] = 1 if sub_choice == "Standard" else 0

        st.markdown("### Contract Length")
        cl_choice = st.radio("Choose one", ["Annual", "Monthly", "Quarterly"],
                             index=1, horizontal=True, key="k_cl_choice")
        v['contract_length_Annual'] = 1 if cl_choice == "Annual" else 0
        v['contract_length_Monthly'] = 1 if cl_choice == "Monthly" else 0
        v['contract_length_Quarterly'] = 1 if cl_choice == "Quarterly" else 0

        predict_clicked = st.form_submit_button("🚀 Predict")

    if predict_clicked:
        return pd.DataFrame([v], columns=FEATURE_ORDER)
    return pd.DataFrame(columns=FEATURE_ORDER)

def score_df(model, df):
    proba = model.predict_proba(df)[:,1] if hasattr(model, "predict_proba") else None
    pred = model.predict(df)
    return pred, proba

def main():
    st.title("Customer Churn Predictor")
    model, err = load_model()
    if err:
        st.error(err)
        st.stop()

    tab1, tab2 = st.tabs(["🔹 Single Prediction", "📦 Batch (CSV)"])

    with tab1:
        df = make_form()
        if not df.empty:
            y_pred, y_proba = score_df(model, df)
            st.success(f"Prediction: {int(y_pred[0])} (1 = churn, 0 = no churn)")
            if y_proba is not None:
                st.info(f"Churn Probability: {y_proba[0]:.4f}")

    with tab2:
        st.write("Upload a CSV that already matches the expected encoded columns:")
        st.code(", ".join(FEATURE_ORDER))
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            missing = [c for c in FEATURE_ORDER if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df = df[FEATURE_ORDER]
                y_pred, y_proba = score_df(model, df)
                df['churn_pred'] = y_pred
                if y_proba is not None:
                    df['churn_proba'] = y_proba
                st.dataframe(df.head(20), use_container_width=True)
                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

if __name__ == "__main__":
    main()
