from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import pickle
import lightgbm as lgb
import base64

app = Flask(__name__)

# Cargar los modelos previamente entrenados
with open('model_prophet.pkl', 'rb') as f:
    model_prophet = pickle.load(f)

lgb_model = lgb.Booster(model_file='lgb_model.txt')

def generar_grafico_linea(fechas, predicciones, producto_seleccionado):
    """Genera un gráfico de línea y lo convierte a base64."""
    plt.figure(figsize=(12, 6))
    plt.plot(fechas, predicciones, label=f'Predicción Ajustada para {producto_seleccionado}', color='blue', marker='o', linestyle='-')
    plt.title(f'Predicción de ventas para el producto {producto_seleccionado}', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Cantidad vendida (Predicción)', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    # Convertir a base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generar_grafico_area(fechas, predicciones, producto_seleccionado):
    """Genera un gráfico de área y lo convierte a base64."""
    plt.figure(figsize=(12, 6))
    plt.fill_between(fechas, predicciones, color='lightgreen', alpha=0.5)
    plt.plot(fechas, predicciones, color='green', linewidth=2)
    plt.title(f'Predicción de Ventas para el Producto: {producto_seleccionado}', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Cantidad Vendida (Predicción)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.7)

    # Convertir a base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generar_grafico_barras(fechas, predicciones, producto_seleccionado):
    """Genera un gráfico de barras y lo convierte a base64."""
    plt.figure(figsize=(12, 6))
    plt.bar(fechas, predicciones, color='skyblue', edgecolor='black')
    plt.title(f'Predicción de Ventas para el Producto: {producto_seleccionado}', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Cantidad Vendida (Predicción)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Convertir a base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        # Obtener datos del usuario
        data = request.json
        producto_seleccionado = data.get('producto')
        fecha_prediccion = data.get('fecha_inicio')
        dias_futuro = int(data.get('dias_futuro', 120))  # Por defecto 120 días

        if not producto_seleccionado or not fecha_prediccion:
            return jsonify({'error': 'Debe proporcionar "producto" y "fecha_inicio"'}), 400

        # Convertir fecha
        fecha_prediccion = pd.to_datetime(fecha_prediccion)

        # Generar fechas futuras
        futuras_fechas = pd.date_range(start=fecha_prediccion, periods=dias_futuro, freq='D')
        futuro_df = pd.DataFrame({'ds': futuras_fechas})
        futuro_df['MES'] = futuro_df['ds'].dt.month
        futuro_df['DIA'] = futuro_df['ds'].dt.day
        futuro_df['AÑO'] = futuro_df['ds'].dt.year
        futuro_df['DIA_SEMANA'] = futuro_df['ds'].dt.weekday
        futuro_df[f'PRODUCTO_{producto_seleccionado}'] = 1

        # Crear columnas faltantes
        X_columns = lgb_model.feature_name()
        for columna in X_columns:
            if columna not in futuro_df.columns:
                futuro_df[columna] = 0

        # Filtrar las columnas necesarias
        X_futuro = futuro_df[X_columns]

        # Predicción con Prophet
        forecast_futuro = model_prophet.predict(futuro_df[['ds']])
        yhat_futuro = forecast_futuro['yhat'].values

        # Ajustar predicción con LightGBM
        predicciones_error_futuro = lgb_model.predict(X_futuro)
        predicciones_futuras_ajustadas = yhat_futuro + predicciones_error_futuro

        # Crear la tabla con las predicciones
        tabla_predicciones = pd.DataFrame({
            'Fecha': futuras_fechas,
            'Predicción Ajustada (Cantidad)': predicciones_futuras_ajustadas
        })

        # Generar gráficos
        grafico_linea = generar_grafico_linea(futuras_fechas, predicciones_futuras_ajustadas, producto_seleccionado)
        grafico_area = generar_grafico_area(futuras_fechas, predicciones_futuras_ajustadas, producto_seleccionado)
        grafico_barras = generar_grafico_barras(futuras_fechas, predicciones_futuras_ajustadas, producto_seleccionado)

        # Devolver los gráficos y la tabla como respuesta en formato JSON
        return jsonify({
            'grafico_linea': grafico_linea,
            'grafico_area': grafico_area,
            'grafico_barras': grafico_barras,
            'tabla': tabla_predicciones.to_dict(orient='records')  # Convertir tabla en JSON
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(host="10.0.11.83", port=5000)
