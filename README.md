# Optimización de Multi-Otsu con Micro Evolución Diferencial Auto-Adaptativa para Segmentación y Clasificación de Neumonía en Radiografías

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

