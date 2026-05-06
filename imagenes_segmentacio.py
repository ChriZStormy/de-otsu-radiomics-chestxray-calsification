import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- FUNCIÓN DE FITNESS EXTERNA ---
def calculate_fitness(thresholds, hist, total_pixels):
    t = np.clip(np.sort(np.round(thresholds).astype(int)), 0, 255)
    D = len(t)
    w = np.zeros(D + 1)
    mu = np.zeros(D + 1)
    bins = np.arange(256)
    
    ranges = [0] + list(t) + [256]
    
    for i in range(D + 1):
        start, end = ranges[i], ranges[i+1]
        if start >= end: continue 
        
        w[i] = np.sum(hist[start:end]) / total_pixels
        if w[i] > 0:
            mu[i] = np.sum(bins[start:end] * hist[start:end]) / (w[i] * total_pixels)
            
    mu_t = np.sum(w * mu)
    sigma_b_sq = np.sum(w * ((mu - mu_t) ** 2))
    return -sigma_b_sq 

# --- CLASE DE OPTIMIZACIÓN: MICRO-DE (µSADE) ---
class uSADE_MultiOtsu:
    def __init__(self, D, NP=10, max_fes=3000, strategy='DE/rand/1', restart_iters=10, e=1, Fl=0.1, Fu=0.9, tau1=0.1, tau2=0.1):
        self.D = D                      
        self.NP = NP                    
        self.max_fes = max_fes 
        self.strategy = strategy
        self.restart_iters = restart_iters 
        self.e = e                      
        self.Fl = Fl                    
        self.Fu = Fu                    
        self.tau1 = tau1
        self.tau2 = tau2

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])
        
        F = np.random.uniform(self.Fl, self.Fl + self.Fu, self.NP)
        Cr = np.random.rand(self.NP)
        
        fes = self.NP 
        t = 1 
        
        while fes < self.max_fes:
            if t % self.restart_iters == 0:
                sort_idx = np.argsort(fitness)
                pop, fitness = pop[sort_idx], fitness[sort_idx]
                F, Cr = F[sort_idx], Cr[sort_idx]
                
                for i in range(self.e):
                    if np.random.rand() < self.tau1: F[i] = self.Fl + np.random.rand() * self.Fu
                    if np.random.rand() < self.tau2: Cr[i] = np.random.rand()
                        
                for i in range(self.e, self.NP):
                    if fes >= self.max_fes: break
                    pop[i] = np.sort(np.random.uniform(0, 255, self.D))
                    fitness[i] = calculate_fitness(pop[i], hist, total_pixels)
                    fes += 1
                    F[i] = np.random.uniform(self.Fl, self.Fl + self.Fu)
                    Cr[i] = np.random.rand()
            
            best_idx = np.argmin(fitness)
            
            for i in range(self.NP):
                if fes >= self.max_fes: break 
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3, r4, r5 = idxs[:5]
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                
                V = np.sort(np.clip(V, 0, 255))
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= Cr[i]) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1 
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
            
            t += 1
                
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int)

# --- RECOLECTORES DE RUTAS ---
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
        if img.suffix.lower() in ['.jpeg', '.jpg'] and "mask" not in str(img).lower():
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

# --- UTILIDADES PARA SEGMENTACIÓN Y GRÁFICOS ---
def get_segmented_image(img, thresholds, D):
    seg = np.zeros_like(img)
    colors = np.linspace(0, 255, D + 1)
    
    seg[img < thresholds[0]] = colors[0]
    for i in range(D - 1):
        mask = (img >= thresholds[i]) & (img < thresholds[i+1])
        seg[mask] = colors[i+1]
    seg[img >= thresholds[-1]] = colors[-1]
    return seg

def plot_and_save_comparison(dataset_name, img_healthy, seg_healthy, img_sick, seg_sick, D):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sano
    axes[0, 0].imshow(img_healthy, cmap='gray')
    axes[0, 0].set_title("Paciente Sano - Imagen Original", fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(seg_healthy, cmap='gray')
    axes[0, 1].set_title(f"Paciente Sano - Segmentación (D={D})", fontsize=14)
    axes[0, 1].axis('off')
    
    # Enfermo
    axes[1, 0].imshow(img_sick, cmap='gray')
    axes[1, 0].set_title(f"Paciente Enfermo ({dataset_name}) - Imagen Original", fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg_sick, cmap='gray')
    axes[1, 1].set_title(f"Paciente Enfermo - Segmentación (D={D})", fontsize=14)
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Comparación de Segmentación usando uSADE (DE/rand/1) - {dataset_name}", fontsize=16, y=0.98)
    plt.tight_layout()
    filename = f"results/images/Plot_Comparativo_{dataset_name}_uSADE_rand_1.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> [GUARDADO] {filename}")

# --- PIPELINE PRINCIPAL ---
def generate_sample_plots(main_path, D=6):
    print("=== RECOPILANDO RUTAS DE IMÁGENES ===")
    problemas = {
        "COVID19": get_covid_paths(main_path),
        "Neumonia": get_neumonia_paths(main_path),
        "Tuberculosis": get_tb_paths(main_path)
    }
    
    optimizer = uSADE_MultiOtsu(D=D, strategy='DE/rand/1', max_fes=D*1000)
    
    for prob_name, image_list in problemas.items():
        if not image_list:
            print(f"![AVISO] No hay imágenes para {prob_name}. Saltando...")
            continue
            
        # Buscar una imagen sana (label 0) y una enferma (label 1)
        path_sano, path_enfermo = None, None
        for img_path, label in image_list:
            if label == 0 and path_sano is None:
                path_sano = img_path
            elif label == 1 and path_enfermo is None:
                path_enfermo = img_path
            
            if path_sano and path_enfermo:
                break
                
        if not path_sano or not path_enfermo:
            print(f"![AVISO] No se encontraron ambas clases (Sano/Enfermo) para {prob_name}. Saltando...")
            continue
            
        print(f"\n[{prob_name}] Procesando imágenes de muestra...")
        
        # Leer imágenes
        img_sano = cv2.imread(str(path_sano), cv2.IMREAD_GRAYSCALE)
        img_enfermo = cv2.imread(str(path_enfermo), cv2.IMREAD_GRAYSCALE)
        
        if img_sano is None or img_enfermo is None:
            print(f"![ERROR] No se pudo leer la imagen de {prob_name}.")
            continue
            
        # Procesar Sano
        print("    Optimizando imagen Sano...")
        th_sano = optimizer.optimize(img_sano)
        seg_sano = get_segmented_image(img_sano, th_sano, D)
        
        # Procesar Enfermo
        print("    Optimizando imagen Enfermo...")
        th_enfermo = optimizer.optimize(img_enfermo)
        seg_enfermo = get_segmented_image(img_enfermo, th_enfermo, D)
        
        # Graficar y guardar
        plot_and_save_comparison(prob_name, img_sano, seg_sano, img_enfermo, seg_enfermo, D)

if __name__ == "__main__": 
    # Asegúrate de tener la carpeta 'dataset' en el mismo directorio
    generate_sample_plots('dataset', D=3)