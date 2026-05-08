import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def procesar_csv(filepath, n_splits=10, n_repeats=4):
    """
    Lee un CSV de extracción radiómica, estandariza y valida con SVM Lineal y Random Forest.
    """
    df = pd.read_csv(filepath)
    if 'label' not in df.columns:
        return None, None
        
    X = df.drop(columns=['label', 'P_0', 'P_1', 'P_2'], errors='ignore').values
    y = df['label'].values
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scaler = StandardScaler()
    
    f1_scores_svm = []
    f1_scores_rf = []
    
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SVM
        svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm.fit(X_train_scaled, y_train)
        y_pred_svm = svm.predict(X_test_scaled)
        f1_scores_svm.append(f1_score(y_test, y_pred_svm, average='weighted'))
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train) # RF usually doesn't need scaling, but it's fine either way
        y_pred_rf = rf.predict(X_test)
        f1_scores_rf.append(f1_score(y_test, y_pred_rf, average='weighted'))
        
    return f1_scores_svm, f1_scores_rf

def generar_tabla_y_estadisticas():
    archivos = glob.glob("results/Radiomics_*.csv")
    
    if not archivos:
        print("[ERROR] No se encontraron archivos en la carpeta results/.")
        return
        
    resultados_svm_dict = {}
    resultados_rf_dict = {}
    f1_por_bloque_svm = []
    
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).replace("Radiomics_", "").replace(".csv", "")
        partes = nombre_base.split("_")
        dataset = partes[0]
        estrategia = "_".join(partes[1:])
        
        f1_scores_svm, f1_scores_rf = procesar_csv(archivo)
        if f1_scores_svm is None: continue
        
        # Guardar para SVM
        if estrategia not in resultados_svm_dict:
            resultados_svm_dict[estrategia] = {}
        resultados_svm_dict[estrategia][dataset] = np.mean(f1_scores_svm)
        
        # Guardar para Random Forest
        if estrategia not in resultados_rf_dict:
            resultados_rf_dict[estrategia] = {}
        resultados_rf_dict[estrategia][dataset] = np.mean(f1_scores_rf)
        
        for fold, f1 in enumerate(f1_scores_svm):
            f1_por_bloque_svm.append({
                'Bloque': f"{dataset}_iter{fold}",
                'Estrategia': estrategia,
                'F1_Score': f1
            })
            
    # --- TABLAS DEL PAPER ---
    df_tabla_svm = pd.DataFrame(resultados_svm_dict).T
    if not df_tabla_svm.empty:
        df_tabla_svm['Promedio'] = df_tabla_svm.mean(axis=1)
        df_tabla_svm = df_tabla_svm.sort_values(by='Promedio', ascending=False)
        print("\n" + "="*60)
        print("TABLA 2 - F1-Score Promedio por Conjunto de Datos y Estrategia (SVM Lineal)")
        print("="*60)
        print(df_tabla_svm.round(4))
        
    df_tabla_rf = pd.DataFrame(resultados_rf_dict).T
    if not df_tabla_rf.empty:
        df_tabla_rf['Promedio'] = df_tabla_rf.mean(axis=1)
        df_tabla_rf = df_tabla_rf.sort_values(by='Promedio', ascending=False)
        print("\n" + "="*60)
        print("RESULTADO ADICIONAL - F1-Score Promedio con Random Forest")
        print("="*60)
        print(df_tabla_rf.round(4))
    
    # --- TEST DE FRIEDMAN (Basado en SVM como en el paper) ---
    df_bloques = pd.DataFrame(f1_por_bloque_svm)
    if df_bloques.empty: return
    
    pivot_df = df_bloques.pivot(index='Bloque', columns='Estrategia', values='F1_Score').dropna()
    estrategias = pivot_df.columns.values
    datos_por_estrategia = [pivot_df[est].values for est in estrategias]
    
    stat, p_value = stats.friedmanchisquare(*datos_por_estrategia)
    
    print("\n" + "="*60)
    print("ANÁLISIS ESTADÍSTICO - PRUEBA DE FRIEDMAN")
    print("="*60)
    print(f"Estadístico Chi-cuadrado: {stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    rangos_friedman = pivot_df.rank(axis=1, ascending=False).mean().sort_values()
    print("\nRanking Promedio de Friedman:")
    print(rangos_friedman.round(4))
    
    if p_value < 0.05:
        print("\n[OK] Hay diferencias significativas. Ejecutando Test Nemenyi...")
        
        # --- TEST POST-HOC NEMENYI ---
        p_values_nemenyi = sp.posthoc_nemenyi_friedman(pivot_df.values)
        p_values_nemenyi.columns = estrategias
        p_values_nemenyi.index = estrategias
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_values_nemenyi, annot=True, cmap="YlGnBu", vmin=0, vmax=0.1, 
                    cbar_kws={'label': 'P-value'}, fmt=".3f", linewidths=0.5)
        plt.title('Test Nemenyi Post-Hoc')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        out_file = "results/heatmap_nemenyi.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"-> Mapa de calor Nemenyi guardado en: {out_file}")
    else:
        print("\n[INFO] No hay diferencias significativas. No se requiere Nemenyi.")
        
def plot_time_bars():
    """Genera un gráfico de barras agrupadas con barras de error para los tiempos de cómputo."""
    time_csv = "results/metrics/Tiempos_Promedio_Por_Imagen.csv"
    if not os.path.exists(time_csv):
        return
        
    df_times = pd.read_csv(time_csv)
    
    plt.figure(figsize=(10, 6))
    
    # Seaborn barplot automatically groups by 'x' and 'hue', but since we already have 
    # the mean and std dev calculated, we need to plot them manually or using pandas plot.
    
    datasets = df_times['Dataset'].unique()
    estrategias = df_times['Estrategia'].unique()
    
    x = np.arange(len(datasets))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for estrategia in estrategias:
        subset = df_times[df_times['Estrategia'] == estrategia]
        # Asegurar orden
        means = []
        stds = []
        for ds in datasets:
            row = subset[subset['Dataset'] == ds]
            if not row.empty:
                means.append(row['Tiempo_Promedio_Segundos'].values[0])
                stds.append(row['Desviacion_Estandar_Segundos'].values[0])
            else:
                means.append(0)
                stds.append(0)
                
        offset = width * multiplier
        rects = ax.bar(x + offset, means, width, label=estrategia, yerr=stds, capsize=5)
        multiplier += 1

    ax.set_ylabel('Tiempo Promedio por Imagen (Segundos)')
    ax.set_title('Costo Computacional por Algoritmo y Dataset (D=3)')
    ax.set_xticks(x + width * (len(estrategias) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(title='Algoritmo')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_file = "results/images/grafico_tiempos.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"\n-> Gráfico de tiempos guardado en: {out_file}")

if __name__ == "__main__":
    os.makedirs('results/images', exist_ok=True)
    generar_tabla_y_estadisticas()
    plot_time_bars()
