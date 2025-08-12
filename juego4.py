import cv2
import pygame
import time
import random
import numpy as np

# ------------------------
# CONFIG
# ------------------------
VIDAS_MAX = 3
TIEMPO_CASTOR = 6                        # s de vida del castor
TIEMPO_CASTOR_MS = TIEMPO_CASTOR * 1000
INTERVALO_CREACION_MS = 600              # espera mínima entre apariciones
ANCHO, ALTO = 1280, 720

# Estabilidad para spawn
VIS_CONSEC_REQ  = 2                      # frames visibles para considerar VISIBLE estable (más ágil)

# Tapado en polígono del QR (rápido)
OCLUSION_ACTIVA        = True
MOTION_RATIO_HIT       = 0.06            # % del polígono con movimiento para tapado (0-1) -> 6%
OCC_WINDOW_MS          = 800             # ventana desde última oclusión mientras era visible
BG_LEARNING_RATE       = 0.001           # estabilidad del fondo (más pequeño = más estable)

# Marcadores disponibles (IDs ArUco)
marcadores_info = {
    1: {"color": "verde",  "pie": "derecho"},
    2: {"color": "verde",  "pie": "derecho"},
    3: {"color": "morado", "pie": "izquierdo"},
    4: {"color": "morado", "pie": "izquierdo"},
}

# ------------------------
# PYGAME
# ------------------------
pygame.init()
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Golpea al Castor - ArUco Edition (rápido + timer rojo)")
fuente = pygame.font.Font(None, 48)
fuente_timer = pygame.font.Font(None, 64)     # un poco más grande
clock = pygame.time.Clock()

castor_verde = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_verde, (30,200,50), castor_verde.get_rect())
castor_morado = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_morado, (150,50,150), castor_morado.get_rect())

# ------------------------
# OPENCV / ARUCO
# ------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam. Revisa dispositivo/camara.")

# Fuerza 1280x720 @30fps (mejora detalle para ArUco)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Crear parámetros (compatibilidad con versiones)
try:
    aruco_params = cv2.aruco.DetectorParameters()
except Exception:
    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except Exception:
        aruco_params = cv2.aruco.DetectorParameters()

# Ajustes para estabilidad (si tu versión los soporta)
try:
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
except Exception:
    pass
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.minMarkerPerimeterRate   = 0.03
aruco_params.maxMarkerPerimeterRate   = 4.0
aruco_params.minCornerDistanceRate    = 0.02

use_detector = False
try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector = True
except Exception:
    detector = None
    use_detector = False
    
USE_RAW_DETECT = True
def detectar_markers(frame_bgr):
   #"""Detecta marcadores ArUco sobre el frame ORIGINAL (sin flip)."""
 if USE_RAW_DETECT:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
 else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

 if use_detector:
        corners, ids, _ = detector.detectMarkers(gray)
 else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
 return corners, ids

def pantalla_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    h, w = frame_rgb.shape[:2]
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
    surf = pygame.transform.scale(surf, (ANCHO, ALTO))
    return surf

# ------------------------
# MÁSCARA DE MOVIMIENTO
# ------------------------
if OCLUSION_ACTIVA:
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
else:
    fgbg = None
    kernel = None

def motion_ratio_in_polygon(fgmask, poly):
    """Porcentaje (0-1) de píxeles en movimiento dentro del polígono (coords de frame)."""
    if fgmask is None or poly is None or len(poly) != 4:
        return 0.0
    mask = np.zeros_like(fgmask, dtype=np.uint8)
    pts = poly.astype(np.int32).reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 255)
    inter = cv2.bitwise_and(fgmask, mask)
    moved = cv2.countNonZero(inter)
    total = cv2.countNonZero(mask)
    if total == 0:
        return 0.0
    return moved / float(total)

# ------------------------
# ESTADO DEL JUEGO
# ------------------------
puntos = 0
vidas  = VIDAS_MAX

# castores activos: {id: {"spawn_ms","pos","color","last_seen_ms","last_occluded_ms","poly"}}
castores = {}
last_create_ms = pygame.time.get_ticks()

# Histéresis para spawn estable
vis_count = {}         # id -> frames consecutivos visible
polys_by_id = {}       # id -> polígono 4x2 (coords frame)

# ------------------------
# BUCLE PRINCIPAL
# ------------------------
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo la cámara")
        break

    # Máscara de movimiento (abrir + dilatar, con learningRate bajo para fondo estable)
    if OCLUSION_ACTIVA:
        fgmask = fgbg.apply(frame, learningRate=BG_LEARNING_RATE)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    else:
        fgmask = None

    # --- Detección ArUco ---
    corners, ids = detectar_markers(frame)

    current_ids = set()
    polys_by_id.clear()

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten().astype(int)
        for i, mid in enumerate(ids_flat):
            current_ids.add(mid)
            vis_count[mid] = vis_count.get(mid, 0) + 1

            # polígono del marcador (coords del frame)
            c = np.array(corners[i]).reshape((-1, 2)).astype(np.float32)  # 4x2
            polys_by_id[mid] = c

        # reset de los no vistos este frame
        for k in list(vis_count.keys()):
            if k not in current_ids:
                vis_count[k] = 0
    else:
        for k in list(vis_count.keys()):
            vis_count[k] = 0

    # Posición en pantalla (centro del polígono reescalado)
    positions_by_id = {}
    for mid, poly in polys_by_id.items():
        cx = int(poly[:,0].mean())
        cy = int(poly[:,1].mean())
        sx = int(cx * (ANCHO / frame.shape[1]))
        sy = int(cy * (ALTO  / frame.shape[0]))
        positions_by_id[mid] = (sx, sy)

    # Estables para spawn
    visible_stable = {m for m, cnt in vis_count.items() if cnt >= VIS_CONSEC_REQ}

    # --- TIEMPO ---
    t_ms = pygame.time.get_ticks()
    despawned_this_frame = False

    # Actualizar estado del castor activo
    for mid, data in list(castores.items()):
        # actualizar pos y polígono si está visible
        if mid in positions_by_id:
            data["pos"] = positions_by_id[mid]
            data["last_seen_ms"] = t_ms
            data["poly"] = polys_by_id.get(mid, data.get("poly", None))
            # registrar oclusión (tapado parcial) mientras está visible
            if OCLUSION_ACTIVA and data["poly"] is not None:
                ratio = motion_ratio_in_polygon(fgmask, data["poly"])
                if ratio >= MOTION_RATIO_HIT:
                    data["last_occluded_ms"] = t_ms

    # 1) Vida por tiempo
    for cid, data in list(castores.items()):
        if (t_ms - data["spawn_ms"]) >= TIEMPO_CASTOR_MS:
            vidas -= 1
            del castores[cid]
            despawned_this_frame = True

    # 2) Puntos por tapado rápido:
    #    Si el marcador NO está visible ahora y hubo oclusión en su polígono "reciente"
    for cid, data in list(castores.items()):
        if cid not in current_ids:
            last_occ = data.get("last_occluded_ms", -10**9)
            last_seen = data.get("last_seen_ms", -10**9)
            # condición: que la desaparición sea cercana a una oclusión (tapado real)
            if (t_ms - last_occ) <= OCC_WINDOW_MS and (t_ms - last_seen) <= OCC_WINDOW_MS:
                # doble chequeo: si aún tenemos su último polígono, mira movimiento en ese polígono en este frame
                ratio_now = motion_ratio_in_polygon(fgmask, data.get("poly", None)) if OCLUSION_ACTIVA else 0.0
                if ratio_now >= (MOTION_RATIO_HIT * 0.7) or (t_ms - last_occ) <= 200:
                    puntos += 1  # golpe válido
            # En cualquier caso, si no se ve, elimínalo para evitar "fantasma"
            del castores[cid]
            despawned_this_frame = True

    # 3) Crear sólo si no hay activo y no acaba de despawnear
    crear_nuevo = (len(castores) == 0) and (not despawned_this_frame)
    if crear_nuevo and (t_ms - last_create_ms) < INTERVALO_CREACION_MS:
        crear_nuevo = False

    if crear_nuevo:
        candidates = [m for m in marcadores_info.keys() if m in visible_stable]
        if candidates:
            mid = random.choice(candidates)
            color = marcadores_info[mid]["color"]
            pos = positions_by_id.get(mid, (ANCHO//2, ALTO//2))
            castores[mid] = {
                "spawn_ms": t_ms,
                "pos": pos,
                "color": color,
                "last_seen_ms": t_ms,
                "last_occluded_ms": -10**9,
                "poly": polys_by_id.get(mid, None)
            }
            last_create_ms = t_ms

    # --- DIBUJO ---
    ventana.fill((0,0,0))
    ventana.blit(pantalla_from_frame(frame), (0,0))

    # Dibujar castor (a lo sumo 1)
    for mid, info in castores.items():
        x, y = info["pos"]
        surf = castor_verde if info["color"] == "verde" else castor_morado
        ventana.blit(surf, (x - surf.get_width()//2, y - surf.get_height()//2))

    # HUD
    ventana.blit(fuente.render(f"Puntos: {puntos}", True, (0,255,0)), (20,20))   # verde
    ventana.blit(fuente.render(f"Vidas: {vidas}",  True, (255,0,0)),   (20,70)) # rojo

    # Cronómetro arriba, con color dinámico (blanco >2s, naranja 2-1s, rojo <=1s)
    if castores:
        cid, data = next(iter(castores.items()))
        remain_ms = max(0, TIEMPO_CASTOR_MS - (t_ms - data["spawn_ms"]))
        remain_sec = (remain_ms + 999) // 1000
        if remain_sec <= 1:
            timer_color = (255, 60, 60)      # rojo
        elif remain_sec <= 2:
            timer_color = (255, 165, 0)      # naranja
        else:
            timer_color = (255, 255, 255)    # blanco
        cron = fuente_timer.render(f"{remain_sec}s", True, timer_color)
        ventana.blit(cron, (ANCHO//2 - cron.get_width()//2, 8))

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

    clock.tick(30)

# limpiar
cap.release()
pygame.quit()
cv2.destroyAllWindows()
