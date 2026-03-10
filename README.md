# Optimización de Multi-Otsu con Micro Evolución Diferencial Auto-Adaptativa para Segmentación y Clasificación de Neumonía en Radiografías 🫁🧬

Este proyecto implementa un *pipeline* avanzado de Visión por Computadora y Machine Learning para la detección de neumonía en imágenes de rayos X de tórax. Utiliza el algoritmo de **Multi-Otsu** optimizado mediante la metaheurística de **Micro Evolución Diferencial Autoadaptativa ($\mu$DE)** para aislar la región de interés (ROI), seguido de extracción de características radiómicas y clasificación mediante Máquinas de Vectores de Soporte (SVM).

## 🚀 Arquitectura del Pipeline

El sistema se divide en dos fases principales, separadas para garantizar la integridad de los datos y prevenir el sesgo (*Data Leakage*).

### Fase 1: Extracción y Generación de Datasets (Generador)
1. **Segmentación Multi-Otsu:** Se divide el histograma de la imagen radiológica en $D+1$ clases (por defecto 4 clases, 3 umbrales).
2. **Optimización $\mu$DE:** Para evitar la explosión combinatoria del algoritmo de fuerza bruta tradicional de Otsu ($O(L^k)$), se utiliza un algoritmo evolutivo con micro-poblaciones ($NP=8$) y parámetros autoadaptativos ($F$, $CR$). Se evalúan 4 esquemas de mutación:
   - `DE/rand/1`
   - `DE/best/1`
   - `DE/rand/2`
   - `DE/best/2`
3. **Extracción Radiómica (Radiomics):** A partir de la máscara binaria generada por el umbral superior (que aísla las opacidades pulmonares), se extraen características:
   - **Morfológicas:** Número de regiones, área máxima, área total.
   - **Textura (GLCM):** Contraste, Energía, Homogeneidad.
4. **Exportación:** Se guardan todas las características extraídas (incluyendo parámetros globales de iluminación) en archivos `.csv` puros. Además, se generan composiciones visuales `.png` de la convergencia y segmentación.

### Fase 2: Clasificación y Evaluación (SVM)
1. **Purga de Sesgo (Ablation):** Antes de entrenar, el script elimina explícitamente las variables dependientes del brillo global de la radiografía ($P_0, P_1, P_2, \mu_2, \sigma_2$) del conjunto de datos. Esto obliga al modelo a aprender de la morfología y textura real del tejido, no de la calibración del equipo de hardware.
2. **Entrenamiento SVM:** Se entrena un clasificador `SVC(kernel='linear')`.
3. **Validación Cruzada:** Se utiliza `StratifiedShuffleSplit` (10 folds, 85% Train / 15% Test) con paralelización vía `joblib` para evaluar simultáneamente los datasets generados por los diferentes esquemas evolutivos.
4. **Métricas:** Se evalúa Accuracy, Recall, F1-Score y AUC, consolidando los resultados en una tabla comparativa final ordenada por exactitud predictiva.

---

## 🧠 Fundamentos Teóricos

Este proyecto fusiona tres dominios complejos de las ciencias computacionales aplicadas a la medicina:

### 1. Multi-Otsu Thresholding
El método de Otsu clásico es un algoritmo de umbralización determinista que busca maximizar la varianza entre clases ($\sigma_B^2$) de un histograma para separar el fondo del objeto. Al extenderlo a múltiples umbrales (Multi-Otsu) para segmentar distintos tipos de tejido pulmonar (hueso, aire, tejido sano, opacidades), la complejidad computacional crece exponencialmente a $O(L^k)$, donde $L$ son los niveles de intensidad y $k$ el número de umbrales. Resolver esto por fuerza bruta en miles de radiografías es inviable.

### 2. Micro-SADE ($\mu$SADE)
Para resolver el cuello de botella de Multi-Otsu, implementamos **Self-Adaptive Micro Differential Evolution**. 
* **Micro ($\mu$):** Utiliza un tamaño de población minúsculo (ej. $NP=8$). Esto permite iteraciones extremadamente rápidas y un bajo consumo de memoria, ideal para procesar miles de imágenes.
* **Auto-adaptativo (SA):** En lugar de configurar manualmente el factor de mutación ($F$) y la tasa de cruza ($CR$), el algoritmo los codifica dentro del individuo. Los parámetros evolucionan dinámicamente junto con la solución, adaptándose automáticamente a la topología del histograma de cada radiografía para evitar la convergencia prematura.

### 3. Radiómica y Análisis GLCM
La radiómica es el proceso de extraer un gran número de características cuantitativas de imágenes médicas. Una vez que $\mu$SADE encuentra los umbrales óptimos, aislamos la región pulmonar de interés. Para describir matemáticamente esta región (y diferenciar una neumonía de un pulmón sano), aplicamos la **Matriz de Co-ocurrencia de Niveles de Gris (GLCM)**. Esta técnica mide la textura espacial de la imagen calculando con qué frecuencia pares de píxeles con valores específicos y en una relación espacial específica ocurren en una imagen, obteniendo descriptores clave como la Energía (uniformidad) y la Homogeneidad.

---

## 🛠️ Tech Stack
* **Lenguaje:** Python
* **Visión por Computadora:** `OpenCV`, `scikit-image`
* **Machine Learning:** `scikit-learn` (SVM, StratifiedShuffleSplit)
* **Optimización Matemática:** `numpy` (Implementación vectorizada de $\mu$DE)
* **Procesamiento Paralelo:** `joblib`

## ⚙️ Instalación y Ejecución
1. Clona el repositorio.
2. Asegúrate de tener las dependencias instaladas: `pip install numpy opencv-python scikit-image scikit-learn pandas joblib matplotlib`
3. Coloca tu dataset de radiografías en la carpeta designada (ej. `/data/xrays/`).
4. Ejecuta el pipeline de la Fase 1 para generar la segmentación y extraer las características CSV.
5. Ejecuta la Fase 2 para entrenar el modelo SVM y visualizar las métricas comparativas de los esquemas evolutivos.

## 📊 Resultados y Desempeño
*(Nota: Inserta aquí una captura de pantalla de las radiografías segmentadas generadas por el script y una tabla o imagen con el Accuracy final de tus modelos SVM comparando los distintos esquemas de mutación DE).*
