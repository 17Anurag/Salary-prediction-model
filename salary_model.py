import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────
# 1. Load & Summarize Dataset
# ─────────────────────────────────────────
df = pd.read_csv("salary_data.csv")

print("=" * 45)
print("       SALARY PREDICTION MODEL")
print("=" * 45)
print("\n📊 Dataset Summary:")
print(f"   Total records   : {len(df)}")
print(f"   Experience range: {df['Experience'].min()} – {df['Experience'].max()} years")
print(f"   Salary range    : ${df['Salary'].min():,} – ${df['Salary'].max():,}")
print(f"   Average salary  : ${df['Salary'].mean():,.2f}")

# ─────────────────────────────────────────
# 2. Train Model
# ─────────────────────────────────────────
X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ─────────────────────────────────────────
# 3. Model Evaluation Metrics
# ─────────────────────────────────────────
r2   = model.score(X_test, y_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📈 Model Performance:")
print(f"   R² Score : {r2:.4f}  (closer to 1.0 is better)")
print(f"   MAE      : ${mae:,.2f}  (avg prediction error)")
print(f"   RMSE     : ${rmse:,.2f}  (penalizes large errors more)")
print(f"\n   Slope (coefficient) : ${model.coef_[0]:,.2f} per year")
print(f"   Intercept           : ${model.intercept_:,.2f}")

# ─────────────────────────────────────────
# 4. User Input & Prediction with Range
# ─────────────────────────────────────────
print("\n" + "─" * 45)
try:
    years = float(input("Enter years of experience: "))
    predicted = model.predict([[years]])[0]
    low  = predicted - rmse
    high = predicted + rmse

    print(f"\n💰 Predicted Salary   : ${predicted:,.2f}")
    print(f"   Likely Range       : ${max(0, low):,.2f} – ${high:,.2f}")
    print(f"   (±${rmse:,.2f} based on model RMSE)")
except ValueError:
    print("Please enter a valid number.")

# ─────────────────────────────────────────
# 5. Visualization — 2 plots side by side
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Salary Prediction Model — Analysis", fontsize=14, fontweight="bold")

# Plot 1: Regression line with confidence band
ax1 = axes[0]
x_range = np.linspace(df["Experience"].min(), df["Experience"].max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

ax1.fill_between(
    x_range.flatten(),
    y_range - rmse,
    y_range + rmse,
    alpha=0.15, color="tomato", label=f"±RMSE band (${rmse:,.0f})"
)
ax1.scatter(df["Experience"], df["Salary"], color="steelblue", zorder=5, label="Actual Data")
ax1.plot(x_range, y_range, color="tomato", linewidth=2, label="Regression Line")
ax1.set_title("Salary vs Experience")
ax1.set_xlabel("Years of Experience")
ax1.set_ylabel("Salary ($)")
ax1.legend()
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Plot 2: Residuals plot
ax2 = axes[1]
all_pred = model.predict(X)
residuals = y - all_pred

ax2.axhline(0, color="tomato", linewidth=1.5, linestyle="--", label="Zero error line")
ax2.scatter(df["Experience"], residuals, color="steelblue", zorder=5)
ax2.set_title("Residuals (Actual − Predicted)")
ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Residual ($)")
ax2.legend()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

plt.tight_layout()
plt.savefig("salary_plot.png", dpi=150)
print("\n📁 Plot saved as salary_plot.png")
plt.show()
