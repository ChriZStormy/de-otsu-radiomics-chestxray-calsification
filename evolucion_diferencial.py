import numpy as np 


def Sphere(x):
    if x.ndim > 1:
        return np.sum(x**2, axis=1)
    else:
        return np.sum(x**2)



class EvolucionDiferencial():
    def __init__(self, NP=10, F=0.5, CR=0.7, limites=[-5, 5], DIM=2, funcion_objetivo=Sphere, GENERACIONES=100):
        self.poblacion = limites[0] + (limites[1] - limites[0]) * np.random.rand(NP, DIM)
        self.fitness_poblacion = np.array(funcion_objetivo(self.poblacion))
        self.mejor_fitness = np.min(self.fitness_poblacion)
        self.historial_mejor_fitness = []
        for i in range(GENERACIONES):
            poblacion_copia = self.poblacion.copy()
            fitness_poblacion_copia = self.fitness_poblacion.copy()
            for ind, individuo in zip(range(len(self.poblacion)), self.poblacion):
                mutante = self._mutacion(self.poblacion, ind, F, limites)
                prueba = self._recombinacion(individuo, mutante, DIM, CR)
                mejor, fitnes = self._seleccion(prueba, individuo, funcion_objetivo)
                poblacion_copia[ind] = mejor
                fitness_poblacion_copia[ind] = fitnes
            self.poblacion = poblacion_copia
            self.fitness_poblacion = fitness_poblacion_copia
            self.historial_mejor_fitness.append(np.min(self.fitness_poblacion))
        self.mejor_fitness = self.historial_mejor_fitness[-1]   

    def _mutacion(self, poblacion, ind, F, limites):
        individuos = []
        indices = [ idx for idx in range(len(poblacion)) if idx != ind]
        indices_individuos = np.random.choice(indices, 3, replace=False)
        mutante = poblacion[indices_individuos[0]] + F * (poblacion[indices_individuos[1]] - poblacion[indices_individuos[2]])
        mutante = np.clip(mutante,  limites[0], limites[1])
        return mutante

    def _recombinacion(self, individuo, mutante, DIM, CR):
        num_cruces = 0
        individuo_prueba = []
        for dimension in range(DIM):
            numero_aleatorio = np.random.random()
            if numero_aleatorio < CR:
                num_cruces += 1
                individuo_prueba.append(mutante[dimension])
            else:
                individuo_prueba.append(individuo[dimension])
        if num_cruces == 0:
            individuo_prueba.pop()
            individuo_prueba.append(mutante[-1])
        return np.array(individuo_prueba)
    
    def _seleccion(self, prueba, individuo, funcion_objetivo):
        fitnes_prueba = funcion_objetivo(prueba)
        fitnes_individuo = funcion_objetivo(individuo)
        if fitnes_prueba < fitnes_individuo:
            return prueba, fitnes_prueba
        else:
            return individuo, fitnes_individuo

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Configuración de parámetros
    DIMENSION = 100
    NP = 30           # Tamaño de población (recomendado 10 * DIM)
    GEN = 2000         # Número de generaciones
    LIMITES = [-5, 5] # Rango de búsqueda
    
    # 2. Instanciar y ejecutar el algoritmo
    print(f"--- Iniciando Evolución Diferencial (Dimensión: {DIMENSION}) ---")
    de = EvolucionDiferencial(
        NP=NP, 
        DIM=DIMENSION, 
        GENERACIONES=GEN, 
        limites=LIMITES,
        F=0.5, 
        CR=0.7
    )
    
    # 3. Mostrar resultados en consola
    print(f"Proceso terminado.")
    print(f"Mejor fitness alcanzado: {de.mejor_fitness:.2e}")
    
    # 4. Generar la gráfica de convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(de.historial_mejor_fitness)), de.historial_mejor_fitness, 
             label='Convergencia DE', color='firebrick', linewidth=2)
    
    # Usamos escala logarítmica en el eje Y porque el fitness se acerca a 0 muy rápido
    plt.yscale('log')
    
    plt.title(f'Curva de Convergencia - Función Sphere ({DIMENSION}D)', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Mejor Fitness (Escala Log)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # El toque final: una anotación del mejor valor
    plt.annotate(f'Mejor: {de.mejor_fitness:.2e}', 
                 xy=(GEN, de.mejor_fitness), 
                 xytext=(GEN*0.7, de.mejor_fitness*100),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.show()