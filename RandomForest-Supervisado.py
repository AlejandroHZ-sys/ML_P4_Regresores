# Random Forest Regression con Iris Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Cargar el dataset Iris

iris = datasets.load_iris()
print("Dataset Iris cargado exitosamente")
print(f"Forma de los datos: {iris.data.shape}")
print(f"Características: {iris.feature_names}")
print(f"Clases: {iris.target_names}")
print()

# Crear DataFrame para mejor manipulación

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Primeras 5 filas del dataset:")
print(df.head())
print()

# Preparar los datos para regresión

# Usaremos las primeras 3 características como X y la 4ta como y (petal width)
X = iris.data[:, :3]  # sepal length, sepal width, petal length
y = iris.data[:, 3]   # petal width (variable a predecir)

print("Variables predictoras (X):")
print("- Sepal Length")
print("- Sepal Width") 
print("- Petal Length")
print()
print("Variable objetivo (y): Petal Width")
print()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")
print()

# Crear y entrenar el modelo Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    criterion='squared_error'
)

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluar el modelo
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

print("=== RESULTADOS DEL MODELO ===")
print(f"R² Score (Entrenamiento): {train_score:.4f}")
print(f"R² Score (Prueba): {test_score:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print()

# Gráfica: valores reales vs predicciones
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred_test, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reales (Petal Width)")
plt.ylabel("Predicciones (Petal Width)")
plt.title("Random Forest Regression - Predicciones vs Valores Reales")
plt.grid(True, alpha=0.3)
plt.show()
