# **Predicción de Lluvia en Australia** 🌦️  

Este proyecto tiene como objetivo predecir la variable `RainTomorrow` (si lloverá o no al día siguiente) utilizando datos climáticos históricos de Australia. Utilizamos diversas técnicas de preprocesamiento, modelos de clasificación y herramientas para el despliegue de aplicaciones.

---

## **Estructura del proyecto**

```plaintext
AA1-TUIA-CALC.../
  ├── docker/
  │   ├── Dockerfile             # Archivo para construir la imagen Docker
  │   ├── inferencia.py          # Script principal de Streamlit para predicciones
  │   ├── mean_encoded_location.pkl  # Archivo de codificación para la variable Location
  │   ├── mediana_evaporation.pkl    # Archivo con datos de mediana imputados para Evaporation
  │   ├── mediana_sunshine.pkl       # Archivo con datos de mediana imputados para Sunshine
  │   ├── medias.pkl                 # Media de las variables para normalización
  │   ├── modelo.pkl                 # Modelo entrenado para la predicción
  │   ├── scaler.pkl                 # Escalador para estandarizar los datos
  ├── Prediccion.csv             # Archivo de ejemplo para validación de predicciones
  ├── README.md                  # Archivo de documentación del proyecto
  ├── requirements.txt           # Dependencias necesarias
  ├── TP-clasificacion-AA1.ipynb # Notebook con el desarrollo completo del trabajo
  ├── weatherAUS.csv             # Dataset original utilizado
```

---

## **Guía de uso**

### **1. Instalación local**

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/Tomas8x/AA1-TUIA-Calcia-Avecilla.git
   cd AA1-TUIA-Calcia-Avecilla
   ```

2. **Crear un entorno virtual (opcional)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar Streamlit**  
   ```bash
   streamlit run docker/inferencia.py
   ```

5. **Subir el archivo `Prediccion.csv` en la interfaz de Streamlit** para verificar predicciones.

---

### **2. Ejecución con Docker**

1. **Construir la imagen Docker**  
   Desde la raíz del proyecto, ejecutar:  
   ```bash
   docker build -t prediccionlluvia .
   ```
1. **Muevete a la carpeta docker**  
   ```bash
   cd AA1-TUIA-Calcia-Avecilla
   cd docker
   ```

2. **Ejecutar el contenedor**  
   ```bash
   docker run -p 8501:8501 prediccionlluvia
   ```

3. **Acceder a la aplicación**  
   Abrir [http://localhost:8501](http://localhost:8501) en tu navegador.

4. **Cargar el archivo `Prediccion.csv` en la interfaz** para obtener resultados.

---

## **Desarrollo del trabajo práctico**

### **Objetivos**

El proyecto cumple con los siguientes objetivos planteados:  

- Implementar modelos de clasificación, optimización de hiperparámetros y métricas (como precisión, recall y F1 score).  
- Entrenar y evaluar una red neuronal con TensorFlow.  
- Analizar la explicabilidad de los modelos usando SHAP.  
- Comparar los resultados entre diferentes modelos de clasificación.  
- Poner el modelo en producción utilizando Streamlit y Docker.

### **Descripción del Dataset**

El dataset utilizado es `weatherAUS.csv`, que contiene información climática histórica de diversas localidades de Australia, incluyendo variables como:
- **Temperatura máxima/mínima**, humedad, presión, lluvia acumulada, etc.  
- **RainToday**: Si llovió o no en el día actual.  
- **RainTomorrow**: Variable objetivo que indica si lloverá o no al día siguiente.  

### **Pasos realizados en el desarrollo**  

#### **1. Preprocesamiento de datos**  
- Selección de 10 ciudades aleatorias del dataset para el análisis.  
- Imputación de valores faltantes usando medianas y otros métodos estadísticos.  
- Codificación de variables categóricas como `Location`.  
- Estandarización de las variables numéricas.  

#### **2. Implementación de modelos**  
- **Modelos base**:  
  - Regresión logística.  
  - Árboles de decisión.  
- **Redes neuronales**:  
  - Arquitectura implementada con TensorFlow.  
- **Optimización de hiperparámetros**:  
  - Grid Search para buscar los mejores valores.  

#### **3. Métricas y análisis**  
- Accuracy, F1-score, recall y matriz de confusión.  
- Curvas ROC para analizar umbrales óptimos.  
- Comparación entre modelos para seleccionar el mejor.  

#### **4. Explicabilidad**  
- SHAP para identificar las variables más importantes y analizar predicciones a nivel global y local.  

#### **5. Producción con MLOps**  
- Despliegue del modelo con Streamlit y Docker.  

---

## **Cómo usar la aplicación**

- **Archivo `Prediccion.csv`**  
  Contiene datos de entrada con las mismas características preprocesadas que el modelo espera. Sube este archivo en la interfaz de Streamlit para obtener predicciones sobre si lloverá (`Yes`) o no (`No`) al día siguiente.

