import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu
import warnings

warnings.filterwarnings("ignore")

class MicroDE_MultiOtsu:
    def __init__(self, NP=8, G_max=50, strategy='DE/rand/1'):
        self.NP = NP
        self.G_max = G_max
        self.strategy = strategy
        self.D = 3  # P0, P1, P2
        
    def fitness_evaluation(self, thresholds, hist, total_pixels):
        """Calcula la varianza negativa (para minimizar) de Multi-Otsu."""
        t = np.clip(np.sort(np.round(thresholds).astype(int)), 0, 255)
        
        w = np.zeros(4)
        mu = np.zeros(4)
        bins = np.arange(256)
        
        w[0] = np.sum(hist[:t[0]]) / total_pixels
        if w[0] > 0: mu[0] = np.sum(bins[:t[0]] * hist[:t[0]]) / (w[0] * total_pixels)
        
        w[1] = np.sum(hist[t[0]:t[1]]) / total_pixels
        if w[1] > 0: mu[1] = np.sum(bins[t[0]:t[1]] * hist[t[0]:t[1]]) / (w[1] * total_pixels)
        
        w[2] = np.sum(hist[t[1]:t[2]]) / total_pixels
        if w[2] > 0: mu[2] = np.sum(bins[t[1]:t[2]] * hist[t[1]:t[2]]) / (w[2] * total_pixels)
        
        w[3] = np.sum(hist[t[2]:]) / total_pixels
        if w[3] > 0: mu[3] = np.sum(bins[t[2]:] * hist[t[2]:]) / (w[3] * total_pixels)
        
        mu_t = np.sum(w * mu)
        sigma_b_sq = np.sum(w * ((mu - mu_t) ** 2))
        
        return -sigma_b_sq

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.random.uniform(0, 255, (self.NP, self.D))
        pop = np.sort(pop, axis=1)
        F = np.random.uniform(0.1, 1.0, self.NP)
        CR = np.random.uniform(0.1, 1.0, self.NP)
        
        fitness = np.array([self.fitness_evaluation(ind, hist, total_pixels) for ind in pop])
        historial_convergencia = []
        
        for g in range(self.G_max):
            best_idx = np.argmin(fitness)
            historial_convergencia.append(fitness[best_idx])
            
            for i in range(self.NP):
                F_i = F[i] if np.random.rand() > 0.1 else np.random.uniform(0.1, 1.0)
                CR_i = CR[i] if np.random.rand() > 0.1 else np.random.uniform(0.1, 1.0)
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3, r4, r5 = idxs[:5]
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F_i * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F_i * (pop[r1] - pop[r2])
                elif self.strategy == 'DE/rand/2':
                    V = pop[r1] + F_i * (pop[r2] - pop[r3] + pop[r4] - pop[r5])
                elif self.strategy == 'DE/best/2':
                    V = pop[best_idx] + F_i * (pop[r1] - pop[r2] + pop[r3] - pop[r4])
                
                V = np.clip(V, 0, 255)
                V = np.sort(V)
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= CR_i) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                f_U = self.fitness_evaluation(U, hist, total_pixels)
                if f_U <= fitness[i]:
                    pop[i] = U
                    fitness[i] = f_U
                    F[i] = F_i
                    CR[i] = CR_i
                    
        best_idx = np.argmin(fitness)
        historial_convergencia.append(fitness[best_idx])
        
        return np.round(pop[best_idx]).astype(int), historial_convergencia

def extract_radiomics(image, thresholds):
    P0, P1, P2 = thresholds
    mu_2 = np.mean(image)
    sigma_2 = np.std(image)
    
    binary_mask = (image >= P2).astype(int)
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    num_regiones = len(regions)
    areas = [r.area for r in regions]
    area_max = max(areas) if areas else 0
    area_total = sum(areas)
    
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contraste = graycoprops(glcm, 'contrast')[0, 0]
    glcm_energia = graycoprops(glcm, 'energy')[0, 0]
    glcm_homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [P0, P1, P2, mu_2, sigma_2, num_regiones, area_max, area_total, glcm_contraste, glcm_energia, glcm_homogeneidad]

def guardar_visualizacion(img, umbrales, estrategia, clase_img, history, out_dir):
    """Genera y guarda la composición visual de la segmentación."""
    img_segmented = np.zeros_like(img)
    img_segmented[img < umbrales[0]] = 0
    img_segmented[(img >= umbrales[0]) & (img < umbrales[1])] = 85
    img_segmented[(img >= umbrales[1]) & (img < umbrales[2])] = 170
    img_segmented[img >= umbrales[2]] = 255
    
    binary_mask = (img >= umbrales[2]).astype(np.uint8) * 255
    
    n_plots = 4 if history else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(15 if history else 12, 4))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original ({clase_img})')
    axes[0].axis('off')
    
    axes[1].imshow(img_segmented, cmap='gray')
    axes[1].set_title(f'Multi-Otsu 4 Clases\nUmbrales: {list(umbrales)}')
    axes[1].axis('off')
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('ROI Radiomics (>= P2)')
    axes[2].axis('off')
    
    if history:
        axes[3].plot(history, color='blue', linewidth=2)
        axes[3].set_title('Convergencia μDE')
        axes[3].set_xlabel('Generación')
        axes[3].set_ylabel('Fitness (-Varianza)')
        axes[3].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    safe_name = estrategia.replace('/', '_')
    plt.savefig(os.path.join(out_dir, f"{safe_name}_{clase_img}.png"), dpi=150)
    plt.close()

def run_pipeline(dataset_path):
    estrategias = ['Standard_Otsu', 'DE/rand/1', 'DE/best/1', 'DE/rand/2', 'DE/best/2']
    base_dir = Path(dataset_path)
    
    dir_resultados = "resultados_segmentacion_umbralizada"
    os.makedirs(dir_resultados, exist_ok=True)
    
    all_images = list(base_dir.rglob("*.jpeg"))
    
    # Instanciar clase estática para evaluar fitness del Standard_Otsu
    evaluador_estatico = MicroDE_MultiOtsu()
    
    for estrategia in estrategias:
        print(f"\n{'-'*60}")
        print(f"[INFO] Procesando con esquema: {estrategia}")
        print(f"{'-'*60}")
        
        safe_name = estrategia.replace('/', '_')
        output_csv = f"radiomics_{safe_name}.csv"
        
        # Banderas para extracción de imágenes de ejemplo
        ejemplo_normal_guardado = False
        ejemplo_neumonia_guardado = False
        
        # Variables de seguimiento global
        mejor_fitness_global = float('inf')
        mejor_individuo_global = None
        
        if os.path.exists(output_csv):
            print(f"[INFO] El archivo '{output_csv}' ya existe. Elimínelo si desea regenerar las imágenes.")
            continue
            
        if estrategia != 'Standard_Otsu':
            optimizer = MicroDE_MultiOtsu(NP=8, G_max=30, strategy=estrategia)
            
        dataset_rows = []
        
        for idx, img_path in enumerate(all_images):
            label_name = img_path.parent.name.upper()
            y_label = 1 if label_name == "PNEUMONIA" else 0
            
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            total_pixels = img.size
            
            # 1. Optimización
            historial = None
            if estrategia == 'Standard_Otsu':
                best_thresholds = threshold_multiotsu(img, classes=4)
            else:
                best_thresholds, historial = optimizer.optimize(img)
            
            # 2. Evaluación de Fitness para registro global
            fitness_actual = evaluador_estatico.fitness_evaluation(best_thresholds, hist, total_pixels)
            if fitness_actual < mejor_fitness_global:
                mejor_fitness_global = fitness_actual
                mejor_individuo_global = best_thresholds
                
            # 3. Guardado de visualizaciones (Solo el primer caso de cada clase)
            if label_name == "NORMAL" and not ejemplo_normal_guardado:
                guardar_visualizacion(img, best_thresholds, estrategia, "NORMAL", historial, dir_resultados)
                ejemplo_normal_guardado = True
                
            elif label_name == "PNEUMONIA" and not ejemplo_neumonia_guardado:
                guardar_visualizacion(img, best_thresholds, estrategia, "PNEUMONIA", historial, dir_resultados)
                ejemplo_neumonia_guardado = True
            
            # 4. Extracción de características
            features = extract_radiomics(img, best_thresholds)
            row = features + [y_label]
            dataset_rows.append(row)
            
            if (idx + 1) % 100 == 0:
                print(f"  -> Procesadas {idx + 1}/{len(all_images)} imágenes...")
                
        # Guardar CSV
        columns = ['P_0', 'P_1', 'P_2', 'mu_2', 'sigma_2', 'num_regiones', 'area_max', 'area_total', 'GLCM_contraste', 'GLCM_energia', 'GLCM_homogeneidad', 'clase']
        df = pd.DataFrame(dataset_rows, columns=columns)
        df.to_csv(output_csv, index=False)
        
        print(f"[COMPLETADO] Dataset guardado: {output_csv}")
        print(f"[GLOBAL BEST] Mejor Fitness (-Varianza): {mejor_fitness_global:.4f}")
        print(f"[GLOBAL BEST] Umbrales del mejor individuo: {mejor_individuo_global}")

if __name__ == "__main__":
    ruta_dataset = 'dataset/chest_xray'
    run_pipeline(ruta_dataset)