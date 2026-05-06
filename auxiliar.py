import os
from pathlib import Path

def contar_clases_neumonia(base_path='dataset'):
    # Ruta base del dataset de neumonía
    p = Path(base_path) / "Chest-X-Ray-Neumonia"
    
    # Contadores
    conteo_normal = 0
    conteo_neumonia = 0
    
    # Verificar si el directorio existe
    if not p.exists():
        print(f"![ERROR] No se encontró el directorio: {p}")
        return

    print(f"Escaneando imágenes en: {p} ...\n")

    # Búsqueda recursiva
    for img in p.rglob("*"):
        if img.is_file() and img.suffix.lower() in ['.jpeg', '.jpg']:
            # Ignorar si la palabra "mask" está en el nombre del archivo
            if "mask" not in str(img).lower():
                ruta_str = str(img).upper()
                
                # Clasificar según el nombre en la ruta
                if "NORMAL" in ruta_str:
                    conteo_normal += 1
                elif "PNEUMONIA" in ruta_str:
                    conteo_neumonia += 1

    # Imprimir resultados
    print("=== RESUMEN DEL DATASET DE NEUMONÍA ===")
    print(f"Normal (Clase 0):    {conteo_normal} instancias")
    print(f"Neumonía (Clase 1):  {conteo_neumonia} instancias")
    print("-" * 39)
    print(f"Total de imágenes:   {conteo_normal + conteo_neumonia}")

if __name__ == "__main__":
    contar_clases_neumonia('dataset')