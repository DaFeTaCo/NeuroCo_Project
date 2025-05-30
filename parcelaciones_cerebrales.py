## EXTRACCIÓN PARCELACIONES CEREBRALES



# %%

from nilearn import datasets

# Cargar el atlas Schaefer con 400 ROIs y 7 redes de Yeo
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)

# El objeto schaefer_atlas es un diccionario-like con información del atlas
atlas_filename = schaefer_atlas['maps'] # Ruta al archivo .nii.gz del atlas
labels = schaefer_atlas['labels'] # Lista de etiquetas de las regiones


# %%
