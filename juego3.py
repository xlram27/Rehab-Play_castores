import cv2
import pygame
import time
import random
import numpy as np

# ------------------------
# CONFIG
# ------------------------
VIDAS_MAX = 3
TIEMPO_CASTOR = 7       # segundos que dura un castor activo
ANCHO = 1280
ALTO  = 720
INTERVALO_CREACION_MS = 1000  # cada cuántos ms intento generar un castor

# marcadores disponibles (IDs generados en tus imágenes ArUco)
marcadores_info = {
    1: {"color": "verde",  "pie": "derecho"},
    2: {"color": "verde",  "pie": "derecho"},
    3: {"color": "morado", "pie": "izquierdo"},
    4: {"color": "morado", "pie": "izquierdo"}
}

# ------------------------
# PYGAME
# ------------------------
pygame.init()
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Golpea al Castor - ArUco Edition")
fuente = pygame.font.Font(None, 48)
clock = pygame.time.Clock()

# Sprites simples (puedes reemplazar con imágenes reales)
castor_verde = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_verde, (30,200,50), castor_verde.get_rect())
castor_morado = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_morado, (150,50,150), castor_morado.get_rect())

# ------------------------
# OPENCV / ARUCO (compatibilidad)
# ------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam. Revisa dispositivo/camara.")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters
try:
    aruco_params = cv2.aruco.DetectorParameters()
except Exception:
    aruco_params = cv2.aruco.DetectorParameters()

# usar ArucoDetector si está disponible (OpenCV 4.7+)
use_detector = False
try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector = True
except Exception:
    detector = None
    use_detector = False

# ------------------------
# ESTADO DEL JUEGO
# ------------------------
puntos = 0
vidas  = VIDAS_MAX

# castores activos: {marker_id: {"spawn": ts, "pos": (x,y), "color":..., "hit": False}}
castores = {}

# visibilidad previa (para detectar transiciones visible->no visible)
prev_visible_ids = set()

# timer para crear castor
last_create_ms = pygame.time.get_ticks()

# ------------------------
# FUNCIONES AUX
# ------------------------
def detectar_markers(frame):
    """Detecta marcadores y devuelve (corners, ids). Compatibilidad new/old API."""
    if use_detector:
        corners, ids, rejected = detector.detectMarkers(frame)
        return corners, ids
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        return corners, ids

def pantalla_from_frame(frame):
    """Convierte frame BGR (cv2) a surface de pygame escalada a ANCHOxALTO.
       Se usa frombuffer para NO rotar la imagen y así mantener consistencia de coordenadas."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)  # required por frombuffer
    h, w = frame_rgb.shape[:2]
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
    surf = pygame.transform.scale(surf, (ANCHO, ALTO))
    return surf

# ------------------------
# BUCLE PRINCIPAL
# ------------------------
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo la cámara")
        break

    # Detectar markers en la imagen original (no escalada) para mejor precisión de corners
    corners, ids = detectar_markers(frame)

    visible_ids = set()
    positions_by_id = {}  # guardamos la (x,y) en coordenadas de pantalla para cada id visible
    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            # Normalizar corners por si vienen en distintos formatos
            c = np.array(corners[i]).reshape((-1, 2))
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            # reescalar a ventana (ANCHO x ALTO)
            sx = int(cx * (ANCHO / frame.shape[1]))
            sy = int(cy * (ALTO  / frame.shape[0]))

            visible_ids.add(mid)
            positions_by_id[mid] = (sx, sy)

            # actualizar la posición del castor si ya existe
            if mid in castores:
                castores[mid]["pos"] = (sx, sy)

    # LOGICA: detectar transicion visible -> no visible (interpretar como pisada)
    now = time.time()
    for mid in list(castores.keys()):
        if (mid in prev_visible_ids) and (mid not in visible_ids):
            # si no fue golpeado ya, contamos el punto
            if not castores[mid]["hit"]:
                castores[mid]["hit"] = True
                puntos += 1
                castores.pop(mid, None)
                continue

    # Crear nuevos castores **solo** si su marcador está visible y usamos la pos exacta
    t_ms = pygame.time.get_ticks()
    if t_ms - last_create_ms > INTERVALO_CREACION_MS:
        last_create_ms = t_ms
        visibles_para_spawn = [m for m in marcadores_info.keys() if m in visible_ids]
        if visibles_para_spawn:
            candidate = random.choice(visibles_para_spawn)
            if candidate not in castores:
                color = marcadores_info[candidate]["color"]
                pos = positions_by_id.get(candidate, (ANCHO//2, ALTO//2))
                # Crear castor exactamente sobre el marcador detectado
                castores[candidate] = {
                    "spawn": time.time(),
                    "pos": pos,
                    "color": color,
                    "hit": False
                }

    # Verificar tiempo de vida de castores (si no se pisan en TIEMPO_CASTOR)
    for mid in list(castores.keys()):
        if time.time() - castores[mid]["spawn"] > TIEMPO_CASTOR:
            vidas -= 1
            castores.pop(mid, None)

    # DIBUJAR
    ventana.fill((0,0,0))
    # fondo cámara (escalado)
    frame_surf = pantalla_from_frame(frame)
    ventana.blit(frame_surf, (0,0))

    # dibujar castores sobre la cámara en sus posiciones
    for mid, info in castores.items():
        x,y = info["pos"]
        surf = castor_verde if info["color"]=="verde" else castor_morado
        ventana.blit(surf, (x - surf.get_width()//2, y - surf.get_height()//2))

    # HUD
    txt_puntos = fuente.render(f"Puntos: {puntos}", True, (255,255,255))
    txt_vidas  = fuente.render(f"Vidas: {vidas}", True, (255,80,80))
    ventana.blit(txt_puntos, (20,20))
    ventana.blit(txt_vidas, (20,70))

    # Game over
    if vidas <= 0:
        go = fuente.render("GAME OVER", True, (255,0,0))
        ventana.blit(go, (ANCHO//2 - go.get_width()//2, ALTO//2 - go.get_height()//2))
        pygame.display.flip()
        time.sleep(2)
        break

    pygame.display.flip()

    # Eventos
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # preparar prev_visible_ids para la siguiente iteración
    prev_visible_ids = visible_ids.copy()

    clock.tick(30)

# limpiar
cap.release()
pygame.quit()
cv2.destroyAllWindows()
