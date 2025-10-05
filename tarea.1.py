import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

datos = {'Superficie_m2': [50, 70, 65, 90, 45], 
         'Num_Habitaciones': [1, 2, 2, 3, 1], 
         'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0], 
         'Precio_UF': [2500, 3800, 3500, 5200, 2100]
         }

df = pd.DataFrame(datos)

X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df[['Precio_UF']]

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Resultados de la evaluacion (Precio UF del Terreno):")
print(f"RMSE: {rmse:.2f} (En promedio, las predicciones se desvian en {rmse:.2f} UF)")
print(f"R-cuadrado (R^2): {r2:.2f} (El {r2*100:.1f}% de la variacion en el precio es explicada por las caracteristicas del terreno)")

