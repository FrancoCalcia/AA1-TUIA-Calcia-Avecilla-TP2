import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Cargar los archivos .pkl
with open("modelo.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("mean_encoded_location.pkl", "rb") as f:
    mean_encoded_location = pickle.load(f)

with open("medias.pkl", "rb") as f:
    medias = pickle.load(f)

with open("mediana_sunshine.pkl", "rb") as f:
    mediana_sunshine = pickle.load(f)

with open("mediana_evaporation.pkl", "rb") as f:
    mediana_evaporation = pickle.load(f)

# Función para preprocesar los datos de entrada
def preprocesar(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = pd.get_dummies(df, columns=['RainToday'], drop_first=True)
    # Renombrar las columnas
    df = df.rename(columns={'RainToday_Yes': 'RainToday'})
    df['RainToday'] = df['RainToday'].astype(int)

    df['Location'] = df['Location'].map(mean_encoded_location)

    for col, media in medias.items():
        if col in df.columns:  
            df[col] = df[col].fillna(media)

    df['Sunshine'] = df['Sunshine'].fillna(mediana_sunshine)
    df['Evaporation'] = df['Evaporation'].fillna(mediana_evaporation)
    # Agregar columna ficticia RainTomorrow con valor 0
    df['RainTomorrow'] = 0  # Valor ficticio solo para que el scaler no falle

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # Lunes = 0, Domingo = 6
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Eliminar columnas innecesarias
    columnas_a_eliminar = ['WindDir9am', 'WindDir3pm', 'WindGustDir', 'Date', 'Pressure9am', 'Temp3pm', 'Temp9am']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])

    # Escalar los datos
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    # Eliminar RainTomorrow escalado ya que no lo necesitas para la predicción
    df_scaled = df_scaled.drop(columns=['RainTomorrow'])
    return df_scaled

# Interfaz de usuario en Streamlit
st.title("Predicción de Lluvia en Australia")

# Subir el archivo CSV con datos
uploaded_file = st.file_uploader("Elija un archivo CSV", type="csv")
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    df_procesado = preprocesar(df_input)

    # Mostrar estadísticas del dataset cargado
    st.write(f"Datos cargados: {df_procesado.shape[0]} filas y {df_procesado.shape[1]} columnas")
    st.write("Estadísticas descriptivas de las variables:")
    st.write(df_procesado.describe())

    # Slider para ajustar el umbral de probabilidad de lluvia
    umbral = st.slider("Selecciona el umbral de probabilidad para la lluvia", 0.0, 1.0, 0.67)

    # Hacer predicción
    prediccion = model.predict(df_procesado)
    
    # Interpretar la predicción según el umbral
    resultados = ["Lloverá" if p >= umbral else "No lloverá" for p in prediccion]
    
    # Mostrar la distribución de las probabilidades de lluvia usando un gráfico de dispersión
    st.write("Distribución de las probabilidades de lluvia")

    fig, ax = plt.subplots(figsize=(15, 9))

    # Asegúrate de que prediccion contenga las probabilidades de lluvia y df_input tenga la columna 'Date'
    ax.scatter(df_input['Date'], prediccion, color='blue', label='Probabilidad de lluvia')

    # Dibuja una línea punteada horizontal en el umbral
    ax.axhline(y=umbral, color='red', linestyle='--', label=f'Umbral de {umbral * 100}%')

    ax.set_title('Distribución de las probabilidades de lluvia')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Probabilidad de Lluvia')
    ax.legend()

    # Mostrar el gráfico
    st.pyplot(fig)

    # Mostrar resultados
    st.write("Predicción de lluvia para el día siguiente:")
    st.write(resultados)

    # Información sobre el modelo utilizado
    st.write("Modelo utilizado para la predicción:")
    st.write("Este modelo es un modelo de Red Neuronal, entrenado y optimizado con datos historicos.")

    # Barra lateral con información adicional
    st.sidebar.markdown("### Acerca de la Aplicación")
    st.sidebar.markdown("Esta aplicación permite predecir si lloverá o no en Australia "
                        "basado en datos históricos de clima. Suba un archivo CSV para obtener los resultados.")
    
    # Botón para descargar los resultados en formato CSV
    st.download_button(
        label="Descargar predicciones",
        data=df_procesado.to_csv(index=False),
        file_name="predicciones_lluvia.csv",
        mime="text/csv"
    )
