## Soluciones

- ✅ Verifica la calidad de los datos sintéticos. Podrían estar dañando el modelo.
- ✅ Elimina features redundantes para evitar colinealidad.
- ✅ Revisa la normalización del precio (quizás solo logaritmo sin MinMaxScaler).
- ✅ Mejora la arquitectura: Reduce dropout, cambia activaciones a LeakyReLU, añade más capas.
- ✅ Prueba Huber Loss o Log-Cosh en lugar de MSE.
- ✅ Ajusta la learning rate (0.0005 en lugar de 0.0015).

## Consideraciones: 
- ⚠️ Verifica el RMSE en Train vs. Test
- ⚠️ Features:

    - Redundancia: Algunas features pueden estar correlacionadas y no aportar nueva información, lo que hace más difícil el entrenamiento.

    - Verifica la colinealidad con un VIF (Variance Inflation Factor).

    - Tipología: El one-hot encoding en una variable binaria puede ser redundante. Usa solo una variable (0 o 1).

    - Ratio Features: ¿Realmente mejoran el modelo? Prueba eliminarlas y evalúa.
- ⚠️ Nuevas features:
- 📌 3. Relación baños/habitaciones (bath_to_room_ratio)
```
df["bath_to_room_ratio"] = df["baths"] / df["rooms"]
df["bath_to_room_ratio"].replace([np.inf, -np.inf], 0, inplace=True)  # Manejo de divisiones por 0
```
- 📌 4. Interacción entre tamaño y número de habitaciones (sqft_rooms_interaction)
```
df["sqft_rooms_interaction"] = df["sqft"] * df["rooms"]
```
- 📌 5. Densidad de viviendas en el vecindario (neighborhood_density)
```
df["neighborhood_density"] = df["neighborhood_price_median"] / df["sqft"]
```
- 📌 6. Factor de lujo (luxury_score)
```
df["luxury_score"] = (df["sqft"] * df["bath_to_room_ratio"]) / df["rooms"]
```
⚠️ **TENER MUY EN CUENTA CORRELACIÓN (< 0.05 eliminar)**