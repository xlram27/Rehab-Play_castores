import cv2
import pygame
import time
import random
import numpy as np
import sys
import os

# --- RUTAS PORTABLES ---
def resource_path(rel):
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)

HIGHSCORE_PATH = resource_path("highscore.txt")

def load_highscore():
    try:
        with open(HIGHSCORE_PATH, "r", encoding="utf-8") as f:
            return int((f.read().strip() or "0"))
    except Exception:
        return 0

def save_highscore(n):
    try:
        with open(HIGHSCORE_PATH, "w", encoding="utf-8") as f:
            f.write(str(int(n)))
    except Exception:
        pass


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
# PYGAME -- inicialización y recursos
# ------------------------
pygame.init()
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Golpea al Castor - ArUco Edition (rápido + timer rojo)")
fuente = pygame.font.Font(None, 48)
fuente_timer = pygame.font.Font(None, 64)
clock = pygame.time.Clock()
start_time_ms = pygame.time.get_ticks()   # cronómetro de la partida

# Audio (después de init)
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.music.set_volume(1.0)

# Ruta del MP3: raíz o carpeta ARCHIVOS
path_sonido = resource_path(os.path.join("ARCHIVOS", "path_sonido.mp3"))  # o "path_sonido.mp3"
if not os.path.exists(path_sonido):
    print("[MISS] No existe el MP3:", path_sonido)
else:
    print("[OK] MP3 listo:", path_sonido)

def play_vida_perdida():
    if not os.path.exists(path_sonido):
        return
    try:
        # Si ya hay algo sonando, córtalo para que se escuche completo
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(path_sonido)
        pygame.mixer.music.play()  # no bloquea
    except Exception as e:
        print("[ERR] Reproduciendo MP3:", e)

# Cargar y escalar CORAZÓN (PNG con transparencia)
heart_candidates = [
    "corazon.png",                          # raíz
    os.path.join("ARCHIVOS", "corazon.png"),# tu carpeta actual
    os.path.join("assets", "corazon.png"),  # por si luego cambias a assets
]
corazon_img = None
for candidate in heart_candidates:
    path = resource_path(candidate)
    if os.path.exists(path):
        try:
            img = pygame.image.load(path).convert_alpha()
            heart_size = max(16, int(ANCHO * 0.035))   # ~3.5% del ancho
            corazon_img = pygame.transform.smoothscale(img, (heart_size, heart_size))
            print(f"[OK] Cargado corazón desde: {path} ({heart_size}px)")
            break
        except Exception as e:
            print(f"[ERR] No pude cargar {path}: {e}")
    else:
        print(f"[MISS] No existe: {path}")

if corazon_img is None:
    print("No se encontró corazon.png. Usaré texto como fallback.")

# Dibujar corazones de vidas (abajo-izquierda)
def dibujar_vidas(surface, vidas, y_offset=10, x_offset=10, spacing=10):
    if corazon_img is not None:
        h = corazon_img.get_height()
        y = ALTO - h - y_offset
        x = x_offset
        for _ in range(max(0, vidas)):
            surface.blit(corazon_img, (x, y))
            x += corazon_img.get_width() + spacing
    else:
        txt = fuente.render(f"Vidas: {vidas}", True, (255, 0, 0))
        surface.blit(txt, (20, 70))

# Sprites simples para castores (puedes reemplazar con imágenes si quieres)
castor_verde = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_verde, (30,200,50), castor_verde.get_rect())
castor_morado = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.ellipse(castor_morado, (150,50,150), castor_morado.get_rect())

# ------------------------
# OPENCV / ARUCO
# ------------------------
def abrir_camara(idx=0, w=1280, h=720):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          30)
            # Warm-up ~0.8 s
            t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t0 < 800:
                pygame.event.pump()
                cap.read()
                pygame.time.delay(30)
            return cap
    return None

cap = abrir_camara(0, 1280, 720)
if cap is None or not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam. Cierra otras apps que usen cámara e intenta de nuevo.")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    aruco_params = cv2.aruco.DetectorParameters()
    
except Exception:
    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except Exception:
        aruco_params = cv2.aruco.DetectorParameters()

# Opcionales para mejorar estabilidad si tu OpenCV los soporta
try:
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
except Exception:
    pass
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.minMarkerPerimeterRate   = 0.03
aruco_params.maxMarkerPerimeterRate   = 4.0
aruco_params.minCornerDistanceRate    = 0.02

# PARCHE 2 - parámetros más permisivos
try:
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
except Exception:
    pass

aruco_params.minMarkerPerimeterRate   = 0.02   # antes 0.03
aruco_params.maxMarkerPerimeterRate   = 4.5
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 35
aruco_params.adaptiveThreshConstant   = 7
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
    """Detecta marcadores ArUco sobre el frame ORIGINAL (sin flip).
       Si USE_RAW_DETECT=True, evita blur/CLAHE para no matar bordes."""
    if USE_RAW_DETECT:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # Compatibilidad con ArucoDetector (OpenCV 4.7+) o detectMarkers clásico
    if use_detector:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids

def pantalla_from_frame(frame):
    """Convierte frame BGR (OpenCV) a surface de pygame escalada a ANCHOxALTO."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    h, w = frame_rgb.shape[:2]
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
    surf = pygame.transform.scale(surf, (ANCHO, ALTO))
    return surf

# ------------------------
# MÁSCARA DE MOVIMIENTO (para detectar tapado/oclusiones)
# ------------------------
if OCLUSION_ACTIVA:
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
else:
    fgbg = None
    kernel = None

def motion_ratio_in_polygon(fgmask, poly):
    """Porcentaje (0-1) de píxeles en movimiento dentro del polígono (coords del frame)."""
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

def game_over_screen(score, elapsed_s, highscore):
    overlay = pygame.Surface((ANCHO, ALTO))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))

    big   = pygame.font.Font(None, 84)
    small = pygame.font.Font(None, 40)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "quit"
                if ev.key == pygame.K_RETURN:
                    return "restart"

        # Dibujo
        ventana.blit(overlay, (0, 0))
        t1 = big.render("GAME OVER", True, (255, 60, 60))
        t2 = small.render(f"Puntaje: {score}", True, (255, 255, 255))
        t3 = small.render(f"Tiempo: {elapsed_s}s", True, (255, 255, 255))
        t4 = small.render(f"Récord: {highscore}", True, (255, 215, 0))
        t5 = small.render("ENTER = jugar de nuevo", True, (200, 200, 200))
        t6 = small.render("ESC = salir", True, (200, 200, 200))

        cx = ANCHO // 2
        ventana.blit(t1, (cx - t1.get_width() // 2, ALTO // 2 - 120))
        ventana.blit(t2, (cx - t2.get_width() // 2, ALTO // 2 - 40))
        ventana.blit(t3, (cx - t3.get_width() // 2, ALTO // 2 + 10))
        ventana.blit(t4, (cx - t4.get_width() // 2, ALTO // 2 + 60))
        ventana.blit(t5, (cx - t5.get_width() // 2, ALTO // 2 + 120))
        ventana.blit(t6, (cx - t6.get_width() // 2, ALTO // 2 + 160))

        pygame.display.flip()
        pygame.time.delay(16)  # ~60 FPS
        

# ------------------------
# ------------------------
# BUCLE PRINCIPAL (REEMPLAZAR TODO ESTE BLOQUE)
# ------------------------
running = True
while running:
    ret, frame = cap.read()
   
    if not ret or frame is None:
        ventana.fill((0,0,0))
        msg = fuente.render("Reconectando cámara...", True, (255,255,255))
        ventana.blit(msg, (ANCHO//2 - msg.get_width()//2, ALTO//2 - msg.get_height()//2))
        pygame.display.flip()
        pygame.event.pump()
        pygame.time.delay(500)  # medio segundo de espera
        
        # Intentar reabrir cámara
        cap.release()
        cap = abrir_camara(0, 1280, 720)  # reutiliza la función del punto B
        
        if cap is None or not cap.isOpened():
            continue  # si no logra abrir, vuelve a intentar
        else:
            # Si se reabrió correctamente, sigue con el bucle
            continue

    if not ret:
        print("Error leyendo la cámara")
        break

    H, W = frame.shape[:2]

    # --- 1) DETECCIÓN en el frame ORIGINAL (sin flip) ---
    corners, ids = detectar_markers(frame)

    # DEBUG opcional: ver marcadores detectados
    # if ids is not None and len(ids) > 0:
    #    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    #else:
    #    cv2.putText(frame, "No ArUco", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

    # --- 2) MÁSCARA DE MOVIMIENTO en el frame ORIGINAL (misma referencia que poly_orig) ---
    if OCLUSION_ACTIVA:
        fgmask = fgbg.apply(frame, learningRate=BG_LEARNING_RATE)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    else:
        fgmask = None

    # --- 3) FRAME PARA MOSTRAR (modo espejo) ---
    frame_display = cv2.flip(frame, 1)

    # --- 4) Construir mapas de polígonos y posiciones ---
    current_ids = set()
    polys_orig = {}   # polígono en coords del frame ORIGINAL (para oclusión)
    polys_disp = {}   # polígono reflejado para DISPLAY (x' = W - x)
    positions_by_id = {}  # centro reflejado -> coords de la ventana pygame

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten().astype(int)
        for i, mid in enumerate(ids_flat):
            current_ids.add(mid)
            # polígono original (4x2) en coords del frame original
            poly_o = np.array(corners[i]).reshape((-1, 2)).astype(np.float32)
            polys_orig[mid] = poly_o

            # polígono reflejado para la imagen mostrada
            poly_d = poly_o.copy()
            poly_d[:, 0] = W - poly_d[:, 0]
            polys_disp[mid] = poly_d

            # centro reflejado -> posición en ventana
            cx = int(poly_d[:, 0].mean())
            cy = int(poly_d[:, 1].mean())
            sx = int(cx * (ANCHO / W))
            sy = int(cy * (ALTO / H))
            positions_by_id[mid] = (sx, sy)

    # --- 5) Visibilidad estable para spawn (mantén tu lógica) ---
    for k in list(vis_count.keys()):
        if k not in current_ids:
            vis_count[k] = 0
    for k in current_ids:
        vis_count[k] = vis_count.get(k, 0) + 1
    visible_stable = {m for m, cnt in vis_count.items() if cnt >= VIS_CONSEC_REQ}

    # --- 6) LÓGICA DE JUEGO ---
    t_ms = pygame.time.get_ticks()
    despawned_this_frame = False

    # Actualizar estado del castor activo
    for mid, data in list(castores.items()):
        # actualizar pos y polígonos
        if mid in positions_by_id:
            data["pos"] = positions_by_id[mid]                 # para dibujar sobre DISPLAY
            data["last_seen_ms"] = t_ms
            data["poly"] = polys_disp.get(mid, data.get("poly", None))   # opcional si lo usas en UI
            data["poly_orig"] = polys_orig.get(mid, data.get("poly_orig", None))  # para oclusión

            # Registrar oclusión SOLO con polígono del frame ORIGINAL
            if OCLUSION_ACTIVA and (data["poly_orig"] is not None):
                ratio = motion_ratio_in_polygon(fgmask, data["poly_orig"])
                if ratio >= MOTION_RATIO_HIT:
                    data["last_occluded_ms"] = t_ms

    # 1) Vida por tiempo
    for cid, data in list(castores.items()):
     if (t_ms - data["spawn_ms"]) >= TIEMPO_CASTOR_MS:
        vidas -= 1
        play_vida_perdida()   # <<< AQUI
        del castores[cid]
        despawned_this_frame = True

    # 2) Puntos por tapado: desaparición + oclusión reciente
    for cid, data in list(castores.items()):
        if cid not in current_ids:
            last_occ = data.get("last_occluded_ms", -10**9)
            last_seen = data.get("last_seen_ms", -10**9)
            if (t_ms - last_occ) <= OCC_WINDOW_MS and (t_ms - last_seen) <= OCC_WINDOW_MS:
                # chequeo rápido adicional usando el último poly_orig
                ratio_now = motion_ratio_in_polygon(fgmask, data.get("poly_orig", None)) if OCLUSION_ACTIVA else 0.0
                if ratio_now >= (MOTION_RATIO_HIT * 0.7) or (t_ms - last_occ) <= 200:
                    puntos += 1
            del castores[cid]
            despawned_this_frame = True

    # 3) Crear solo si no hay activo y no acaba de despawnear
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
                "poly": polys_disp.get(mid, None),       # para UI si lo quieres
                "poly_orig": polys_orig.get(mid, None)   # para oclusión/movimiento
            }
            last_create_ms = t_ms

    # --- 7) DIBUJO (usa frame_display) ---
    ventana.fill((0,0,0))
    ventana.blit(pantalla_from_frame(frame_display), (0,0))

    # Castor(es) sobre el video
    for mid, info in castores.items():
       x, y = info["pos"]
       surf = castor_verde if info["color"] == "verde" else castor_morado
       ventana.blit(surf, (x - surf.get_width()//2, y - surf.get_height()//2))

    # HUD: puntos + cronómetro + vidas (al final)
    ventana.blit(fuente.render(f"Puntos: {puntos}", True, (0,255,0)), (20,20))

    if castores:
        cid, data = next(iter(castores.items()))
        remain_ms = max(0, TIEMPO_CASTOR_MS - (t_ms - data["spawn_ms"]))
        remain_sec = (remain_ms + 999) // 1000
        timer_color = (255,255,255) if remain_sec > 2 else ((255,165,0) if remain_sec > 1 else (255,60,60))
        cron = fuente_timer.render(f"{remain_sec}s", True, timer_color)
        ventana.blit(cron, (ANCHO//2 - cron.get_width()//2, 8))

    dibujar_vidas(ventana, vidas)

    pygame.display.flip()      # <- ¡imprescindible!
    for ev in pygame.event.get():
      if ev.type == pygame.QUIT:
        running = False
      elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
        running = False
    clock.tick(30)

    if vidas <= 0:
    # 1) tiempo jugado
     elapsed_s = (pygame.time.get_ticks() - start_time_ms) // 1000

    # 2) récord
     hs = load_highscore()
     if puntos > hs:
        save_highscore(puntos)
        hs = puntos

    # 3) mostrar pantalla y esperar acción
     action = game_over_screen(puntos, elapsed_s, hs)

    # 4) manejar acción
     if action == "restart":
        puntos = 0
        vidas  = VIDAS_MAX
        castores.clear()
        vis_count.clear()
        last_create_ms = pygame.time.get_ticks()
        start_time_ms  = pygame.time.get_ticks()  # reinicia cronómetro
        continue
     else:  # "quit"
        running = False
        break

cap.release()
pygame.quit()
cv2.destroyAllWindows()
sys.exit()
