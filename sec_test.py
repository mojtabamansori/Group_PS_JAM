import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری داده‌ها
file_path = "DATASET/ANN-GTU-104E-102.xlsx"  # مسیر فایل را اگر لازم است تغییر دهید
data = pd.read_excel(file_path)

# انتخاب ویژگی‌ها (تمام ستون‌ها به جز ستون‌های 12 و 13) و هدف (ستون 12 برای خروجی، ستون 13 به عنوان هدف بهینه)
X = data.iloc[:, :-2].values  # ویژگی‌ها
y_output = data.iloc[:, -2].values  # ستون زرد (خروجی پیش‌بینی شده)
y_optimal = data.iloc[:, -1].values  # ستون قرمز (هدف بهینه)

# تقسیم داده‌ها برای آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y_output, test_size=0.2, random_state=42)

# تعریف مدل جنگل تصادفی
model = RandomForestRegressor(n_estimators=100, random_state=42)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی بر روی مجموعه تست
y_pred = model.predict(X_test)

# ارزیابی مدل
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# مقایسه با ستون بهینه (قرمز) در صورت نیاز
optimal_mse = mean_squared_error(y_test, y_optimal[:len(y_test)])
optimal_r2 = r2_score(y_test, y_optimal[:len(y_test)])

print(f"Optimal Column MSE: {optimal_mse}")
print(f"Optimal Column R^2: {optimal_r2}")

# ذخیره نمودار پیش‌بینی‌ها و مقادیر واقعی
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig("predicted_vs_actual.png")  # ذخیره به عنوان فایل تصویری
plt.close()  # بسته کردن نمودار برای جلوگیری از نمایش آن

# ذخیره نمودار مقایسه MSE و R^2
plt.figure(figsize=(10, 6))
metrics = ['MSE', 'R^2']
values = [mse, r2]
optimal_values = [optimal_mse, optimal_r2]
plt.bar(metrics, values, alpha=0.6, label='Model')
plt.bar(metrics, optimal_values, alpha=0.4, label='Optimal Column')
plt.title('Model vs Optimal Column Performance')
plt.ylabel('Score')
plt.legend()
plt.savefig("model_vs_optimal.png")  # ذخیره به عنوان فایل تصویری
plt.close()

# ذخیره نمودار اهمیت ویژگی‌ها
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
plt.xticks(range(len(feature_importances)), sorted_indices + 1)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig("feature_importances.png")  # ذخیره به عنوان فایل تصویری
plt.close()

# مرحله 2: بهینه‌سازی ورودی‌ها برای نزدیک شدن به ستون قرمز (هدف بهینه)
# تعریف تابع هدف برای بهینه‌سازی
def optimization_objective(X_input, model, target_optimal):
    """
    هدف: کمینه کردن اختلاف خروجی مدل با مقدار اپتیمال (ستون قرمز)
    """
    X_input = np.array(X_input).reshape(1, -1)  # تبدیل ورودی به فرمت مناسب
    y_pred = model.predict(X_input)
    return np.abs(y_pred - target_optimal)

# نمونه اولیه (برای ورودی‌های یک نمونه از X_test استفاده می‌کنیم)
initial_sample = X_test[0]  # استفاده از ایندکس برای آرایه‌های NumPy

# مقدار اپتیمال مربوط به این نمونه
target_optimal = y_optimal[0]  # مقدار ستون قرمز

# محدودیت‌های بهینه‌سازی (مقدار حداقل و حداکثر برای هر ویژگی)
bounds = [(X_train.min(axis=0)[col], X_train.max(axis=0)[col]) for col in range(X_train.shape[1])]

# اجرای بهینه‌سازی
result = minimize(
    optimization_objective,
    x0=initial_sample,
    args=(model, target_optimal),
    bounds=bounds,
    method='L-BFGS-B'
)

# نمایش نتیجه بهینه‌سازی
optimized_inputs = result.x
optimized_output = model.predict(optimized_inputs.reshape(1, -1))[0]

print("\nOptimized Inputs:")
print(optimized_inputs)
print(f"Optimized Output (Yellow): {optimized_output:.4f}, Target (Red): {target_optimal:.4f}")

# ذخیره نمودار مقایسه ورودی‌های بهینه‌سازی شده با هدف بهینه
plt.figure(figsize=(10, 6))
plt.plot(range(len(optimized_inputs)), optimized_inputs, label='Optimized Inputs', marker='o')
plt.axhline(y=target_optimal, color='red', linestyle='--', label='Target Optimal')
plt.title('Optimized Inputs vs Target Optimal')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.legend()
plt.savefig("optimized_inputs_vs_target.png")  # ذخیره به عنوان فایل تصویری
plt.close()
