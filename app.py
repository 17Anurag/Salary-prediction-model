import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💰 Salary Prediction Model")
st.markdown("Predict your estimated salary based on years of experience using **Linear Regression**.")

# ─────────────────────────────────────────
# Load & Train
# ─────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("salary_data.csv")
    X = df[["Experience"]]
    y = df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = model.score(X_test, y_test)
    return df, model, rmse, mae, r2

df, model, rmse, mae, r2 = load_and_train()

# ─────────────────────────────────────────
# Dataset Preview
# ─────────────────────────────────────────
with st.expander("📂 View Dataset"):
    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────
# Model Metrics
# ─────────────────────────────────────────
st.subheader("📈 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"${mae:,.2f}")
col3.metric("RMSE", f"${rmse:,.2f}")

# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────
st.subheader("🔮 Predict Your Salary")
years = st.slider("Years of Experience", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

predicted = model.predict([[years]])[0]
low  = max(0, predicted - rmse)
high = predicted + rmse

st.success(f"Estimated Salary: **${predicted:,.2f}**")
st.caption(f"Likely range: ${low:,.2f} — ${high:,.2f}  (±RMSE of ${rmse:,.2f})")

# ─────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────
st.subheader("📊 Regression Chart")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_range = np.linspace(df["Experience"].min(), df["Experience"].max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

# Plot 1: Regression line
ax1 = axes[0]
ax1.fill_between(x_range.flatten(), y_range - rmse, y_range + rmse,
                 alpha=0.15, color="tomato", label=f"±RMSE band")
ax1.scatter(df["Experience"], df["Salary"], color="steelblue", zorder=5, label="Actual Data")
ax1.plot(x_range, y_range, color="tomato", linewidth=2, label="Regression Line")
ax1.axvline(x=years, color="green", linestyle="--", linewidth=1.5, label=f"Your input ({years} yrs)")
ax1.set_title("Salary vs Experience")
ax1.set_xlabel("Years of Experience")
ax1.set_ylabel("Salary ($)")
ax1.legend(fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Plot 2: Residuals
ax2 = axes[1]
residuals = df["Salary"] - model.predict(df[["Experience"]])
ax2.axhline(0, color="tomato", linewidth=1.5, linestyle="--")
ax2.scatter(df["Experience"], residuals, color="steelblue", zorder=5)
ax2.set_title("Residuals Plot")
ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Residual ($)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

plt.tight_layout()
st.pyplot(fig)
