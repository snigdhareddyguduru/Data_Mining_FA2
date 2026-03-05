import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

st.title("ATM Intelligence Demand Forecasting Dashboard")

# Load dataset
df = pd.read_csv("atm_transactions.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------
# Exploratory Data Analysis
# -------------------------

st.header("Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.histplot(df["Total_Withdrawals"], bins=20, ax=ax)
ax.set_title("Distribution of Withdrawals")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x=df["Total_Withdrawals"], ax=ax)
ax.set_title("Outliers in Withdrawals")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.barplot(x="Day_of_Week", y="Total_Withdrawals", data=df, ax=ax)
ax.set_title("Withdrawals by Day of Week")
st.pyplot(fig)

# -------------------------
# Clustering
# -------------------------

st.header("ATM Demand Clustering")

features = df[["Total_Withdrawals", "Total_Deposits"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3)
df["Cluster"] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots()

sns.scatterplot(
    x=df["Total_Withdrawals"],
    y=df["Total_Deposits"],
    hue=df["Cluster"],
    palette="Set2",
    ax=ax
)

ax.set_title("ATM Demand Clusters")
st.pyplot(fig)

# -------------------------
# Anomaly Detection
# -------------------------

st.header("Anomaly Detection")

df["zscore"] = zscore(df["Total_Withdrawals"])
df["Anomaly"] = df["zscore"].abs() > 2

fig, ax = plt.subplots()

colors = df["Anomaly"].map({True: "red", False: "blue"})

ax.scatter(df["Total_Withdrawals"], df["Total_Deposits"], c=colors)

ax.set_title("Unusual Withdrawal Activity")
st.pyplot(fig)

st.write("Red points indicate abnormal withdrawal spikes.")

# -------------------------
# Interactive Planner
# -------------------------

st.header("ATM Demand Filter")

day = st.selectbox("Choose Day of Week", df["Day_of_Week"].unique())

filtered = df[df["Day_of_Week"] == day]

st.write("Filtered Transactions")
st.write(filtered)

st.write(
    "Average Withdrawals:",
    filtered["Total_Withdrawals"].mean()
)