import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Importar nilearn para visualización de neuroimagen
from nilearn import plotting
from nilearn import datasets
from nilearn import image # Necesario para cargar imágenes NIfTI si es un fallback

# --- 1. Cargar las matrices de correlación ---
# Asume que los archivos CSV están en el mismo directorio que este script.
# Si no, especifica la ruta completa a los archivos.
file_names = [
    "AD_mean_correlation_matrix.csv",
    "CNAD_mean_correlation_matrix.csv",
    "CNMCI_mean_correlation_matrix.csv",
    "MCI_mean_correlation_matrix.csv"
]

correlation_matrices = {}

# Imprimir el directorio de trabajo actual para depuración
current_directory = os.getcwd()
complemento = "NeuroCo_Project/Pruebas/DATA/"
current_directory = os.path.join(current_directory,complemento)## correcciones locales 
print(f"Directorio de trabajo actual: {current_directory}")

for file_name in file_names:
    # Construir la ruta completa al archivo
    full_file_path = os.path.join(current_directory, file_name)
    print(f"Intentando cargar el archivo: {full_file_path}")

    try:
        # Extraer el nombre del grupo del nombre del archivo (ej. "AD", "CNAD")
        group_name = file_name.split('_')[0]
        # Cargar la matriz de correlación, asumiendo que es un CSV sin encabezado ni índice
        matrix = pd.read_csv(full_file_path, header=None, index_col=None).values
        correlation_matrices[group_name] = matrix
        print(f"Matriz '{file_name}' cargada para el grupo: {group_name}")
        print(f"Dimensiones de la matriz: {matrix.shape}")
    except FileNotFoundError:
        print(f"Error: El archivo '{full_file_path}' no se encontró. Asegúrate de que esté en la ruta correcta o que el script se ejecute desde el directorio correcto.")
    except Exception as e:
        print(f"Ocurrió un error al cargar '{full_file_path}': {e}")

# --- 2. Extraer grafos y plotearlos ---
# Para extraer los grafos, necesitamos un umbral para binarizar la matriz de correlación.
# Las correlaciones por debajo del umbral se considerarán como no-conexiones.
# Un umbral común es 0.4, pero puede variar según el estudio.
threshold = 0.4 # Puedes ajustar este umbral

# --- Preparar datos del atlas Schaefer para la visualización ---
# Descargar el atlas Schaefer 2018 con 200 parcelas y 7 redes.
# Si tus matrices corresponden a un número diferente de parcelas,
# ajusta 'n_rois' y 'yeo_networks' según sea necesario.
print("\nDescargando atlas Schaefer para obtener coordenadas de ROIs...")
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)

# --- DEBUGGING: Imprimir las claves disponibles en schaefer_atlas ---
print(f"Claves disponibles en schaefer_atlas: {schaefer_atlas.keys()}")

# Las coordenadas de los centroides de las ROIs
# FIX: Intentar usar 'coords' primero, si no existe, calcularlas desde el mapa del atlas.
node_coords = None
if 'coords' in schaefer_atlas:
    node_coords = schaefer_atlas['coords']
    print("Usando la clave 'coords' para las coordenadas de los nodos.")
else:
    print("Advertencia: La clave 'coords' no se encontró en schaefer_atlas. Intentando calcular los centroides desde el mapa del atlas.")
    try:
        # plotting.find_parcellation_cut_coords puede extraer las coordenadas de un archivo NIfTI
        node_coords = plotting.find_parcellation_cut_coords(schaefer_atlas['maps'])
        print("Centroides calculados exitosamente desde el mapa del atlas.")
    except Exception as e:
        print(f"Error al calcular los centroides desde el mapa del atlas: {e}")
        print("No se pudieron obtener las coordenadas de los nodos. La visualización de conectomas no será posible.")
        node_coords = None # Asegurarse de que node_coords sea None si falla

if node_coords is None:
    print("No se pudieron obtener las coordenadas de los nodos. Saltando la sección de ploteo de conectomas.")
else:
    print(f"Coordenadas de ROIs del atlas Schaefer cargadas. Número de ROIs: {len(node_coords)}")

    plt.figure(figsize=(15, 10))

    # Verificar si se cargaron matrices antes de intentar plotear
    if not correlation_matrices:
        print("No se cargaron matrices de correlación. No se pueden plotear los grafos.")
    else:
        # Plotear cada grafo superpuesto al atlas
        # Se ajusta el número de subplots dinámicamente según el número de matrices cargadas
        num_matrices = len(correlation_matrices)
        rows = int(np.ceil(num_matrices / 2))
        cols = 2 if num_matrices > 1 else 1 # Asegura al menos 1 columna si solo hay 1 grafo

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), subplot_kw={'projection': '3d'})
        # Aplanar el array de ejes solo si hay más de 1 subplot para evitar errores con axes[i]
        if num_matrices > 1:
            axes = axes.flatten()
        else:
            axes = [axes] # Convertir a lista para que sea iterable de la misma forma

        for i, (group_name, matrix) in enumerate(correlation_matrices.items()):
            # Para la visualización de conectomas, es mejor usar la matriz de correlación original
            # y aplicar el umbral para las aristas, en lugar de una matriz binarizada.
            # Las aristas representarán la fuerza de la correlación (peso).
            # Asegurarse de que la matriz tenga el mismo número de nodos que las coordenadas del atlas.
            if matrix.shape[0] != len(node_coords):
                print(f"Advertencia: El número de nodos en la matriz '{group_name}' ({matrix.shape[0]}) no coincide con el número de ROIs del atlas Schaefer ({len(node_coords)}). La visualización podría ser incorrecta.")
                # Si hay un desajuste, no ploteamos este grafo pero continuamos con los demás
                continue

            # Aplicar el umbral a la matriz de correlación para definir las conexiones a plotear.
            # Los valores por debajo del umbral se pondrán a cero.
            display_matrix = matrix.copy()
            display_matrix[np.abs(display_matrix) < threshold] = 0

            # Crear el plot_connectome en el eje actual
            plotting.plot_connectome(display_matrix, node_coords,
                                     edge_threshold=threshold, # Umbral para mostrar aristas
                                     display_mode='lzr', # Vista de cerebro de cristal
                                     title=f'Conectividad funcional para {group_name} (Umbral={threshold})',
                                     axes=axes[i], # Asignar a un subplot específico
                                     node_size=20, # Tamaño de los nodos
                                     edge_kwargs={'linewidth': 1, 'alpha': 0.7}) # Estilo de las aristas

        plt.tight_layout()
        plt.show()

# --- 3. Aplicar métricas de integración y segregación funcional ---
# Elegiremos 3 métricas principales:
# Integración:
# 1. Longitud de camino promedio (Average Path Length): Mide la eficiencia de la transferencia de información en la red.
# 2. Eficiencia global (Global Efficiency): Una medida alternativa de integración que no requiere que la red esté conectada.
# Segregación:
# 3. Coeficiente de agrupamiento promedio (Average Clustering Coefficient): Mide la tendencia de los nodos a agruparse.

print("\n--- Métricas de Integración y Segregación Funcional ---")
if not correlation_matrices:
    print("No se cargaron matrices de correlación. No se pueden calcular las métricas.")
else:
    for group_name, matrix in correlation_matrices.items():
        # Para las métricas, se usa la matriz binarizada
        adj_matrix = (np.abs(matrix) > threshold).astype(int)
        G = nx.from_numpy_array(adj_matrix)
        G.remove_edges_from(nx.selfloop_edges(G))

        print(f"\nGrupo: {group_name}")

        # Métrica de Integración 1: Longitud de camino promedio
        # Solo es aplicable si el grafo está conectado.
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"  Longitud de Camino Promedio (Average Path Length): {avg_path_length:.4f}")
        else:
            print("  El grafo no está conectado, no se puede calcular la Longitud de Camino Promedio directamente.")
            # Para grafos desconectados, se puede calcular para cada componente conectado
            # o usar la eficiencia global como una alternativa más robusta.

        # Métrica de Integración 2: Eficiencia Global
        # Esta métrica es más robusta para grafos desconectados.
        global_efficiency = nx.global_efficiency(G)
        print(f"  Eficiencia Global (Global Efficiency): {global_efficiency:.4f}")

        # Métrica de Segregación 3: Coeficiente de Agrupamiento Promedio
        avg_clustering_coefficient = nx.average_clustering(G)
        print(f"  Coeficiente de Agrupamiento Promedio (Average Clustering Coefficient): {avg_clustering_coefficient:.4f}")
