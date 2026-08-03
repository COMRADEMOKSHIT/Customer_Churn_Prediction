import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Telco Customer Churn Explorer",
    layout="wide",
)

st.title("📊 Telco Customer Churn — EDA Dashboard")
st.caption("Upload the Telco churn CSV (or use the bundled sample) to explore it.")


@st.cache_data
def load_data(file) -> pd.DataFrame:
    data = pd.read_csv(file)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()
    if "customerID" in data.columns:
        data = data.drop(["customerID"], axis=1)
    return data


uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


if uploaded_file is not None:
    df = load_data(uploaded_file)
elif st.sidebar.checkbox("Use bundled sample file (if present)", value=True):
    try:
        df = load_data(DEFAULT_PATH)
    except FileNotFoundError:
        st.warning(
            f"Couldn't find `{DEFAULT_PATH}` next to the app. "
            "Upload a CSV in the sidebar to get started."
        )
        st.stop()
else:
    st.info("Upload a CSV in the sidebar to get started.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Column types**")
    st.dataframe(df.dtypes.astype(str).rename("dtype"))
with col2:
    st.markdown("**Missing values (after cleaning)**")
    st.dataframe(df.isnull().sum().rename("nulls"))


st.subheader("Categorical feature distributions")

sns.set_theme(style="ticks", palette="Set2", color_codes=True)
palette = "Set2"

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
categorical_cols = [c for c in categorical_cols if c in df.columns]

n_cols = 5
n_rows = int(np.ceil(len(categorical_cols) / n_cols))

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 5 * n_rows))
axes = np.array(axes).reshape(n_rows, n_cols)

for idx, colname in enumerate(categorical_cols):
    r, c = divmod(idx, n_cols)
    ax = sns.countplot(x=colname, data=df, ax=axes[r][c], palette=palette)
    if colname == "PaymentMethod":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for idx in range(len(categorical_cols), n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r][c].axis("off")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Numeric feature distributions")

numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]

fig2, axes2 = plt.subplots(len(numeric_cols), 1, figsize=(8, 3 * len(numeric_cols)))
if len(numeric_cols) == 1:
    axes2 = [axes2]

colors = ["b", "r", "g"]
for i, colname in enumerate(numeric_cols):
    sns.kdeplot(df[colname], fill=True, color=colors[i % len(colors)], ax=axes2[i])
    axes2[i].set_title(colname)

fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)


st.subheader("Label-encoded data (preview)")


def object_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        series = LabelEncoder().fit_transform(series)
    return series


encoded_df = df.apply(lambda col: object_to_int(col))
st.dataframe(encoded_df.head())

st.download_button(
    "Download encoded data as CSV",
    data=encoded_df.to_csv(index=False).encode("utf-8"),
    file_name="telco_churn_encoded.csv",
    mime="text/csv",
)
