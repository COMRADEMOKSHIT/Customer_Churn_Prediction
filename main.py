import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
st.read(df)
df.head()

df.dtypes

df.isnull().sum()

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()

df = df.dropna()
df.isnull().sum()

df = df.drop(['customerID'], axis = 1)
df.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", palette="Set2", color_codes=True)
palette = "Set2"

fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))
sns.countplot(x = "gender", data = df, ax=axes[0][0], palette=palette)
sns.countplot(x = "Partner", data = df, ax=axes[0][1], palette=palette)
sns.countplot(x = "Dependents", data = df, ax=axes[0][2], palette=palette)
sns.countplot(x = "PhoneService", data = df, ax=axes[0][3], palette=palette)
sns.countplot(x = "MultipleLines", data = df, ax=axes[0][4], palette=palette)
sns.countplot(x = "InternetService", data = df, ax=axes[1][0], palette=palette)
sns.countplot(x = "OnlineSecurity", data = df, ax=axes[1][1], palette=palette)
sns.countplot(x = "OnlineBackup", data = df, ax=axes[1][2], palette=palette)
sns.countplot(x = "DeviceProtection", data = df, ax=axes[1][3], palette=palette)
sns.countplot(x = "TechSupport", data = df, ax=axes[1][4], palette=palette)
sns.countplot(x = "StreamingTV", data = df, ax=axes[2][0], palette=palette)
sns.countplot(x = "StreamingMovies", data = df, ax=axes[2][1], palette=palette)
sns.countplot(x = "Contract", data = df, ax=axes[2][2], palette=palette)
sns.countplot(x = "PaperlessBilling", data = df, ax=axes[2][3], palette=palette)
ax = sns.countplot(x = "PaymentMethod", data = df, ax=axes[2][4], palette=palette)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
sns.kdeplot(df["tenure"], shade=True, color="b",ax = ax1)
sns.kdeplot(df["MonthlyCharges"], shade=True, color="r", ax = ax2)
sns.kdeplot(df["TotalCharges"], shade=True, color="g", ax = ax3)
fig.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

df = df.apply(lambda x: object_to_int(x))
df.head()
