import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

def calculate_fitness(thresholds, hist, total_pixels):
    """Calcula la varianza negativa (para minimizar) de Multi-Otsu D=3 (4 regiones)."""
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

class uSADE_MultiOtsu:
    def __init__(self, NP=10, max_fes=3000, strategy='DE/best/1', restart_iters=10):
        self.NP = NP
        self.max_fes = max_fes
        self.strategy = strategy
        self.restart_iters = restart_iters
        self.D = 3
        self.e = 1 # Elitismo

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])
        
        F = np.random.uniform(0.1, 0.9, self.NP)
        CR = np.random.rand(self.NP)
        
        fes = self.NP
        t = 1
        
        convergence = []
        
        while fes < self.max_fes:
            if t % self.restart_iters == 0:
                sort_idx = np.argsort(fitness)
                pop, fitness = pop[sort_idx], fitness[sort_idx]
                F, CR = F[sort_idx], CR[sort_idx]
                
                # Elitismo: mantenemos los self.e mejores
                for i in range(self.e, self.NP):
                    if fes >= self.max_fes: break
                    pop[i] = np.sort(np.random.uniform(0, 255, self.D))
                    fitness[i] = calculate_fitness(pop[i], hist, total_pixels)
                    fes += 1
                    F[i] = np.random.uniform(0.1, 0.9)
                    CR[i] = np.random.rand()
            
            best_idx = np.argmin(fitness)
            convergence.append(fitness[best_idx])
            
            for i in range(self.NP):
                if fes >= self.max_fes: break
                
                # Self-adaptation
                if np.random.rand() < 0.1: F[i] = np.random.uniform(0.1, 0.9)
                if np.random.rand() < 0.1: CR[i] = np.random.rand()
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3 = idxs[:3]
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                
                V = np.sort(np.clip(V, 0, 255))
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= CR[i]) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                U = np.sort(U)
                
                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
                    
            t += 1
            
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int), convergence

class StandardDE_MultiOtsu:
    def __init__(self, NP=15, max_fes=3000, strategy='DE/rand/1'):
        self.NP = NP
        self.max_fes = max_fes
        self.strategy = strategy
        self.D = 3
        self.F = 0.5
        self.CR = 0.9

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])
        
        fes = self.NP
        convergence = []
        
        while fes < self.max_fes:
            best_idx = np.argmin(fitness)
            convergence.append(fitness[best_idx])
            
            for i in range(self.NP):
                if fes >= self.max_fes: break
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3 = idxs[:3]
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + self.F * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + self.F * (pop[r1] - pop[r2])
                
                V = np.sort(np.clip(V, 0, 255))
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= self.CR) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                U = np.sort(U)
                
                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
                    
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int), convergence

def extract_radiomics(image, thresholds):
    P0, P1, P2 = thresholds
    mu_2 = np.mean(image)
    sigma_2 = np.std(image)
    
    # Análisis Morfológico (en ROI >= P2 que suele ser consolidación/hueso)
    binary_mask = (image >= P2).astype(np.uint8)
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    num_regiones = len(regions)
    areas = [r.area for r in regions]
    area_max = max(areas) if areas else 0
    area_total = sum(areas)
    
    # Análisis Textura GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contraste = graycoprops(glcm, 'contrast')[0, 0]
    glcm_energia = graycoprops(glcm, 'energy')[0, 0]
    glcm_homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [P0, P1, P2, mu_2, sigma_2, num_regiones, area_max, area_total, glcm_contraste, glcm_energia, glcm_homogeneidad]

# --- RUTAS DE DATASETS ---
def get_covid_paths(base):
    paths = []
    p = Path(base) / "Chest-X-Ray-COVID19"
    for folder, label_val in [("Normal", 0), ("COVID", 1)]:
        target_dir = p / folder / "images"
        if target_dir.exists():
            for img in target_dir.glob("*"):
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg'] and "mask" not in str(img).lower():
                    paths.append((img, label_val))
    return paths

def get_neumonia_paths(base):
    paths = []
    p = Path(base) / "Chest-X-Ray-Neumonia"
    for img in p.rglob("*"):
        if img.suffix.lower() in ['.jpeg', '.jpg', '.png'] and "mask" not in str(img).lower():
            if img.is_file():
                if "NORMAL" in str(img).upper(): paths.append((img, 0))
                elif "PNEUMONIA" in str(img).upper(): paths.append((img, 1))
    return paths

def get_tb_paths(base):
    paths = []
    p = Path(base) / "Chest-X-Ray-Tuberculosis"
    csv_path = p / "MetaData.csv"
    if not csv_path.exists(): return []

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_id = str(row['id'])
        label_val = int(row['ptb']) 
        img_name = f"{img_id}.png" 
        img_path = p / "Chest-X-Ray" / "image" / img_name
        if not img_path.exists():
            img_path = p / "images" / img_name
        if img_path.exists() and "mask" not in str(img_path).lower():
            paths.append((img_path, label_val))
    return paths

# --- PROCESAMIENTO PARALELO ---
def procesar_combinacion(dataset_name, image_list, estrategia, max_fes):
    safe_name = estrategia.replace('/', '_')
    output_csv = f"results/Radiomics_{dataset_name}_{safe_name}.csv"
    
    if os.path.exists(output_csv):
        print(f" -> [OMITIENDO] {output_csv} ya existe.")
        # Retornamos None si se omitió
        return None
        
    print(f" -> [INICIANDO] {dataset_name} | {estrategia}")
    start_time = time.time()
    
    if estrategia.startswith('uSADE'):
        opt_strat = 'DE/best/1' if 'best' in estrategia else 'DE/rand/1'
        optimizer = uSADE_MultiOtsu(NP=10, max_fes=max_fes, strategy=opt_strat)
    elif estrategia.startswith('DE'):
        opt_strat = 'DE/best/1' if 'best' in estrategia else 'DE/rand/1'
        optimizer = StandardDE_MultiOtsu(NP=15, max_fes=max_fes, strategy=opt_strat)
    
    dataset_rows = []
    image_times = []
    
    for img_path, label_val in image_list:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        total_pixels = img.size
        
        if estrategia == 'Standard_Otsu':
            try:
                start_img = time.time()
                best_thresholds = threshold_multiotsu(img, classes=4)
                end_img = time.time()
                image_times.append(end_img - start_img)
            except:
                continue
        else:
            start_img = time.time()
            best_thresholds, _ = optimizer.optimize(img)
            end_img = time.time()
            image_times.append(end_img - start_img)
            
        features = extract_radiomics(img, best_thresholds)
        row = features + [label_val]
        dataset_rows.append(row)
        
    if dataset_rows:
        columns = ['P_0', 'P_1', 'P_2', 'mu_2', 'sigma_2', 'num_regiones', 'area_max', 'area_total', 'GLCM_contraste', 'GLCM_energia', 'GLCM_homogeneidad', 'label']
        df = pd.DataFrame(dataset_rows, columns=columns)
        df.to_csv(output_csv, index=False)
        
    elapsed = time.time() - start_time
    
    mean_time = np.mean(image_times) if image_times else 0
    std_time = np.std(image_times) if image_times else 0
    
    print(f" <- [FINALIZADO] {dataset_name} - {estrategia} | Tiempo: {elapsed/60:.2f} mins | Promedio/img: {mean_time:.4f}s (±{std_time:.4f})")
    
    return {
        'Dataset': dataset_name,
        'Estrategia': estrategia,
        'Tiempo_Promedio_Segundos': mean_time,
        'Desviacion_Estandar_Segundos': std_time
    }

def run_pipeline(base_path):
    os.makedirs('results', exist_ok=True)
    
    problemas = {
        "COVID19": get_covid_paths(base_path),
        "Neumonia": get_neumonia_paths(base_path),
        "Tuberculosis": get_tb_paths(base_path)
    }
    
    estrategias = ['Standard_Otsu', 'DE_rand_1', 'DE_best_1', 'uSADE_rand_1', 'uSADE_best_1']
    max_fes = 3000 # Para D=3
    
    tareas = []
    for dataset_name, image_list in problemas.items():
        if not image_list:
            print(f"![AVISO] No se encontraron imágenes para {dataset_name}")
            continue
        for est in estrategias:
            tareas.append((dataset_name, image_list, est, max_fes))
            
    print(f"=== INICIANDO EXTRACCIÓN PARALELA ({len(tareas)} tareas) ===")
    timing_results = Parallel(n_jobs=-1)(
        delayed(procesar_combinacion)(d_name, i_list, est, fes) for d_name, i_list, est, fes in tareas
    )
    
    timing_results = [t for t in timing_results if t is not None]
    if timing_results:
        df_times = pd.DataFrame(timing_results)
        df_times.to_csv("results/metrics/Tiempos_Promedio_Por_Imagen.csv", index=False)
        print("\n[OK] Tiempos guardados en results/metrics/Tiempos_Promedio_Por_Imagen.csv")
        
    print("=== PIPELINE COMPLETADO ===")

if __name__ == "__main__":
    run_pipeline('dataset')