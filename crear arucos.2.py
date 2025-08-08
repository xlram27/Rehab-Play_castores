import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# Carpeta de salida
output_dir = Path("marcadores_aruco")
output_dir.mkdir(exist_ok=True)

# Diccionario ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Lista de IDs para los marcadores
marcadores = {
    "CASTOR_VERDE_PIE_DERECHO_1": 1,
    "CASTOR_VERDE_PIE_DERECHO_2": 2,
    "CASTOR_MORADO_PIE_IZQUIERDO_1": 3,
    "CASTOR_MORADO_PIE_IZQUIERDO_2": 4
}

# Tamaño del marcador en píxeles
marker_size = 400
borde_blanco = 20  # píxeles de margen

for nombre, marker_id in marcadores.items():
    # Generar el marcador como matriz NumPy
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Convertir a imagen PIL
    pil_img = Image.fromarray(marker_img)
    pil_img = pil_img.convert("RGB")

    # Agregar borde blanco
    pil_img_con_borde = ImageOps.expand(pil_img, border=borde_blanco, fill="white")

    # Guardar imagen
    pil_img_con_borde.save(output_dir / f"{nombre}.png")

print("✅ Marcadores con bordes blancos generados en:", output_dir)

