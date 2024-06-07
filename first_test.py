import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


file_path = r'DATASET/PART1.xlsx'
df = pd.read_excel(file_path)
df = np.array(df)

x = df[:, 0:3]
y = df[:, 3]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = RandomForestRegressor()
mesal = np.array([23.91756821, 7.58775806, 7.58749294])
model.fit(X_train, y_train)
mesal = np.array([25, 12, 11])
mesal = mesal.reshape(1, -1)
y_pred = model.predict(mesal)
print(y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')
