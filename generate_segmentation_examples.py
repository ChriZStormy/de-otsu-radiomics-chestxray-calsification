import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datasets_controller as dc
from skimage.filters import threshold_multiotsu

def apply_thresholds(image, thresholds):
    """Aplica los umbrales a la imagen para segmentarla en regiones."""
    # thresholds debe tener longitud 3 (para 4 clases)
    t = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(image)
    segmented[image < t[0]] = 0
    segmented[(image >= t[0]) & (image < t[1])] = 85
    segmented[(image >= t[1]) & (image < t[2])] = 170
    segmented[image >= t[2]] = 255
    return segmented

def plot_segmentations(img, results, dataset_name, condition, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original: {dataset_name} ({condition})')
    axes[0].axis('off')
    
    # Métodos
    methods = ['Standard_Otsu', 'uSADE_best_1', 'uSADE_rand_1', 'DE_best_1', 'DE_rand_1']
    
    for i, m_name in enumerate(methods, start=1):
        thresholds = results.get(m_name, [0, 0, 0])
        seg_img = apply_thresholds(img, thresholds)
        
        axes[i].imshow(seg_img, cmap='viridis')
        clean_thresholds = [int(x) for x in thresholds]
        axes[i].set_title(f'{m_name}\nT={clean_thresholds}')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    output_dir = "results/segmentation_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando rutas de datasets...")
    base_path = 'dataset'
    datasets = {
        "COVID19": dc.get_covid_paths(base_path),
        "Neumonia": dc.get_neumonia_paths(base_path),
        "Tuberculosis": dc.get_tb_paths(base_path)
    }
    
    # Seleccionar 1 sano (0) y 1 infectado (1) por cada dataset
    selected_images = {}
    for name, paths in datasets.items():
        if not paths:
            print(f"Advertencia: No se encontraron imágenes para {name}")
            continue
            
        try:
            normal_img = next(p[0] for p in paths if p[1] == 0)
            infected_img = next(p[0] for p in paths if p[1] == 1)
            selected_images[name] = {
                "Sano": normal_img, 
                "Infectado": infected_img
            }
        except StopIteration:
            print(f"Advertencia: No se pudo encontrar una imagen sana y una infectada para {name}")
            
    # Algoritmos a evaluar (Max_FES configurado a 3000 según paper)
    max_fes = 3000
    
    for dataset_name, conditions in selected_images.items():
        for condition, img_path in conditions.items():
            print(f"Procesando: {dataset_name} - {condition}")
            
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error al cargar la imagen: {img_path}")
                continue
                
            results = {}
            
            # 1. Standard Otsu
            try:
                print("  -> Ejecutando Standard Otsu")
                otsu_thresholds = threshold_multiotsu(img, classes=4)
                results["Standard_Otsu"] = np.round(otsu_thresholds).astype(int)
            except Exception as e:
                print(f"  -> Error en Otsu: {e}")
                results["Standard_Otsu"] = [0, 0, 0]
                
            # 2. uSADE best/1
            print("  -> Ejecutando uSADE best/1")
            usade_best = dc.uSADE_MultiOtsu(NP=10, max_fes=max_fes, strategy='DE/best/1')
            th, _ = usade_best.optimize(img)
            results["uSADE_best_1"] = th
            
            # 3. uSADE rand/1
            print("  -> Ejecutando uSADE rand/1")
            usade_rand = dc.uSADE_MultiOtsu(NP=10, max_fes=max_fes, strategy='DE/rand/1')
            th, _ = usade_rand.optimize(img)
            results["uSADE_rand_1"] = th
            
            # 4. DE best/1
            print("  -> Ejecutando DE best/1")
            de_best = dc.StandardDE_MultiOtsu(NP=15, max_fes=max_fes, strategy='DE/best/1')
            th, _ = de_best.optimize(img)
            results["DE_best_1"] = th
            
            # 5. DE rand/1
            print("  -> Ejecutando DE rand/1")
            de_rand = dc.StandardDE_MultiOtsu(NP=15, max_fes=max_fes, strategy='DE/rand/1')
            th, _ = de_rand.optimize(img)
            results["DE_rand_1"] = th
            
            # Generar plot
            save_path = os.path.join(output_dir, f"segmentation_{dataset_name}_{condition}.png")
            plot_segmentations(img, results, dataset_name, condition, save_path)
            print(f"  -> Guardado en: {save_path}")

if __name__ == "__main__":
    main()
