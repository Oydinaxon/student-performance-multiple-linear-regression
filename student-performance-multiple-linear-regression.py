import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# ======================
# 1. DATASET
# ======================
np.random.seed(42)

n = 50

study_hours = np.random.randint(1, 10, n)
attendance = np.random.randint(50, 100, n)
homework = np.random.randint(40, 100, n)

# Sun'iy, lekin real jarayonga yaqin formula
grades = 5 * study_hours + 0.3 * attendance + 0.2 * homework + np.random.randint(-5, 5, n)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "homework": homework,
    "grade": grades
})

print("Datasetning dastlabki 5 qatori:")
print(df.head())

# ======================
# 2. MODEL
# ======================
X = df[["study_hours", "attendance", "homework"]]
y = df["grade"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# ======================
# 3. TENGLAMA
# ======================
b0 = model.intercept_
b1, b2, b3 = model.coef_

print("\nHosil bo‘lgan ko‘p omilli regressiya tenglamasi:")
print(f"y = {b0:.2f} + {b1:.2f}*x1 + {b2:.2f}*x2 + {b3:.2f}*x3")

print("\nBu yerda:")
print("x1 = study_hours")
print("x2 = attendance")
print("x3 = homework")

# ======================
# 4. MODEL BAHOLASH
# ======================
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("\nModel sifati:")
print(f"R2 = {r2:.4f}")
print(f"MSE = {mse:.4f}")

# ======================
# 5. REAL VS PREDICTED
# ======================
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred)
plt.xlabel("Haqiqiy baho")
plt.ylabel("Prognoz baho")
plt.title("Haqiqiy va prognoz baholar taqqoslanishi")
plt.grid(True)
plt.show()

# ======================
# 6. KOEFFITSIENTLAR GRAFIKI
# ======================
features = ["Study Hours", "Attendance", "Homework"]

plt.figure(figsize=(8, 6))
plt.bar(features, model.coef_)
plt.xlabel("Omillar")
plt.ylabel("Koeffitsient qiymati")
plt.title("Omillarning regressiya modelidagi ta’siri")
plt.grid(True)
plt.show()

# ======================
# 7. 3D REGRESSIYA TEKISLIGI
# ======================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Haqiqiy nuqtalar
ax.scatter(df["study_hours"], df["attendance"], df["grade"], label="Haqiqiy ma'lumotlar")

# Tekislik uchun grid
x_surf, y_surf = np.meshgrid(
    np.linspace(df["study_hours"].min(), df["study_hours"].max(), 20),
    np.linspace(df["attendance"].min(), df["attendance"].max(), 20)
)

# homework o‘rtacha qiymatda olinadi
mean_homework = df["homework"].mean()

# Regressiya tekisligi
z_surf = b0 + b1 * x_surf + b2 * y_surf + b3 * mean_homework

ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5)

ax.set_xlabel("O‘qish soati")
ax.set_ylabel("Qatnashish")
ax.set_zlabel("Baho")
ax.set_title("Ko‘p omilli regressiya tekisligi")

plt.show()
