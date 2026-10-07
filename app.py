import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tumor Predictor", page_icon="🔬", layout="wide")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


art = load()
model, features = art["model"], art["features"]

st.title("🔬 Breast Tumor Prediction")
st.caption(f"Random Forest model · test accuracy {art['accuracy']:.1%} · demo only, not medical advice")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV upload)"])

with tab_single:
    st.sidebar.header("Input features")
    # Show the 10 "mean" features as sliders; the rest use dataset medians
    shown = [f for f in features if f.startswith("mean")]
    values = dict(art["defaults"])
    for f in shown:
        lo, hi = float(art["mins"][f]), float(art["maxs"][f])
        values[f] = st.sidebar.slider(f, lo, hi, float(art["defaults"][f]))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[features]
        proba = model.predict_proba(row)[0]
        pred = int(proba.argmax())
        label = art["target_names"][pred]

        c1, c2 = st.columns(2)
        c1.metric("Prediction", label.capitalize())
        c2.metric("Confidence", f"{proba[pred]:.1%}")
        st.bar_chart(pd.Series(proba, index=art["target_names"]))
    else:
        st.info("Adjust the sliders in the sidebar, then click **Predict**.")

with tab_batch:
    st.write("Upload a CSV containing these columns:")
    st.code(", ".join(features), language=None)
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = set(features) - set(df.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            out = df.copy()
            out["prediction"] = [art["target_names"][i] for i in model.predict(df[features])]
            st.dataframe(out)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
