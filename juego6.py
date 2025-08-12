import os, sys, time, random
import cv2, pygame, numpy as np

# ========= Utilidades / Rutas =========
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

def load_highscore(p=resource_path("highscore.txt")):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return int(f.read().strip() or "0")
    except:
        return 0

def save_highscore(n, p=resource_path("highscore.txt")):
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(int(n)))
    except:
        pass

def load_image(rel, size=None):
    path = resource_path(rel)
    if not os.path.exists(path):
        return None
    try:
        surf = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(surf, size) if size else surf
    except:
        return None

def play_mp3_once(mp3_path):
    if not os.path.exists(mp3_path):
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
    except:
        pass

# ========= Config =========
ANCHO, ALTO = 1280, 720
VIDAS_MAX = 3
TIEMPO_CASTOR_MS = 6000
INTERVALO_CREACION_MS = 600
VIS_CONSEC_REQ = 2

OCLUSION_ACTIVA = True
MOTION_RATIO_HIT = 0.06
OCC_WINDOW_MS = 800
BG_LEARNING_RATE = 0.001

MARCADORES = {1: "verde", 2: "verde", 3: "morado", 4: "morado"}

# ========= Pygame =========
pygame.init()
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Golpea al Castor - ArUco (lite)")
fuente = pygame.font.Font(None, 48)
fuente_timer = pygame.font.Font(None, 64)
clock = pygame.time.Clock()
start_time_ms = pygame.time.get_ticks()

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.music.set_volume(1.0)
SND_PERDIDA = resource_path(os.path.join("ARCHIVOS", "path_sonido.mp3"))

heart = load_image(os.path.join("ARCHIVOS", "corazon.png"),
                   (int(ANCHO*0.035), int(ANCHO*0.035)))
castor_verde  = load_image(os.path.join("ARCHIVOS","castor_verde.png"),
                           (int(ANCHO*0.35), int(ANCHO*0.35)))
castor_morado = load_image(os.path.join("ARCHIVOS","castor_morado.png"),
                           (int(ANCHO*0.35), int(ANCHO*0.35)))

# Fallbacks simples si faltan imágenes
if castor_verde is None:
    castor_verde = pygame.Surface((120,120), pygame.SRCALPHA)
    pygame.draw.ellipse(castor_verde,(30,200,50),castor_verde.get_rect())
if castor_morado is None:
    castor_morado = pygame.Surface((120,120), pygame.SRCALPHA)
    pygame.draw.ellipse(castor_morado,(150,50,150),castor_morado.get_rect())

def dibujar_vidas(surface, vidas):
    if heart:
        x, y = 10, ALTO - heart.get_height() - 10
        for _ in range(max(0, vidas)):
            surface.blit(heart, (x, y))
            x += heart.get_width() + 10
    else:
        surface.blit(fuente.render(f"Vidas: {vidas}", True, (255,0,0)), (20,70))

def pantalla_from_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(
        np.ascontiguousarray(frame_rgb).tobytes(),
        frame_rgb.shape[1::-1],
        "RGB"
    )
    return pygame.transform.scale(surf, (ANCHO, ALTO))

# ========= Cámara (robusta + warm-up) =========
def abrir_camara(idx=0, w=1280, h=720):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          30)
            t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t0 < 800:
                pygame.event.pump()
                cap.read()
                pygame.time.delay(30)
            return cap
    return None

cap = abrir_camara(0, ANCHO, ALTO)
if cap is None:
    raise RuntimeError("No se pudo abrir la webcam. Cierra otras apps que la usen e intenta de nuevo.")

# ========= ArUco =========
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    aruco_params = cv2.aruco.DetectorParameters()
except Exception:
    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except Exception:
        aruco_params = cv2.aruco.DetectorParameters()

try: aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
except: pass
aruco_params.minMarkerPerimeterRate = 0.02
aruco_params.maxMarkerPerimeterRate = 4.5
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 35
aruco_params.adaptiveThreshConstant  = 7
aruco_params.minCornerDistanceRate   = 0.02

try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector = True
except Exception:
    detector = None
    use_detector = False

def detectar_markers(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if use_detector:
        corners, ids, _ = detector.detectMarkers(gray)  # el "_" ignora el tercero
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids

# ========= Movimiento / oclusión =========
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False) if OCLUSION_ACTIVA else None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) if OCLUSION_ACTIVA else None

def motion_ratio_in_polygon(fgmask, poly):
    if fgmask is None or poly is None or len(poly) != 4:
        return 0.0
    mask = np.zeros_like(fgmask, np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32).reshape((-1,1,2))], 255)
    inter = cv2.bitwise_and(fgmask, mask)
    t = cv2.countNonZero(mask)
    return 0.0 if t == 0 else cv2.countNonZero(inter) / float(t)

# ========= Game Over =========
def game_over_screen(score, elapsed_s, highscore):
    overlay = pygame.Surface((ANCHO, ALTO)); overlay.set_alpha(200); overlay.fill((0,0,0))
    big, small = pygame.font.Font(None, 84), pygame.font.Font(None, 40)
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:   return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return "quit"
                if ev.key == pygame.K_RETURN: return "restart"

        ventana.blit(overlay, (0,0))
        t1 = big.render("GAME OVER", True, (255,60,60))
        t2 = small.render(f"Puntaje: {score}", True, (255,255,255))
        t3 = small.render(f"Tiempo: {elapsed_s}s", True, (255,255,255))
        t4 = small.render(f"Récord: {highscore}", True, (255,215,0))
        t5 = small.render("ENTER = jugar de nuevo", True, (200,200,200))
        t6 = small.render("ESC = salir", True, (200,200,200))
        cx = ANCHO//2
        ventana.blit(t1, (cx - t1.get_width()//2, ALTO//2 - 120))
        ventana.blit(t2, (cx - t2.get_width()//2, ALTO//2 - 40))
        ventana.blit(t3, (cx - t3.get_width()//2, ALTO//2 + 10))
        ventana.blit(t4, (cx - t4.get_width()//2, ALTO//2 + 60))
        ventana.blit(t5, (cx - t5.get_width()//2, ALTO//2 + 120))
        ventana.blit(t6, (cx - t6.get_width()//2, ALTO//2 + 160))
        pygame.display.flip()
        pygame.time.delay(16)

# ========= Estado del juego =========
puntos, vidas = 0, VIDAS_MAX
castores = {}                 # {id: {...}} solo habrá 1 activo
last_create_ms = pygame.time.get_ticks()
vis_count = {}                # id -> frames consecutivos visible

# ========= Bucle principal =========
running = True
while running:
    ret, frame = cap.read()
    if not ret or frame is None:
        ventana.fill((0,0,0))
        msg = fuente.render("Reconectando cámara...", True, (255,255,255))
        ventana.blit(msg, (ANCHO//2 - msg.get_width()//2, ALTO//2 - msg.get_height()//2))
        pygame.display.flip(); pygame.event.pump(); pygame.time.delay(500)
        cap.release(); cap = abrir_camara(0, ANCHO, ALTO)
        continue

    H, W = frame.shape[:2]
    corners, ids = detectar_markers(frame)

    if OCLUSION_ACTIVA:
        fgmask = fgbg.apply(frame, learningRate=BG_LEARNING_RATE)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    else:
        fgmask = None

    frame_display = cv2.flip(frame, 1)

    current_ids = set()
    polys_orig, polys_disp, positions_by_id = {}, {}, {}

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten().astype(int)
        for i, mid in enumerate(ids_flat):
            current_ids.add(mid)
            poly_o = np.array(corners[i]).reshape((-1,2)).astype(np.float32)
            polys_orig[mid] = poly_o
            poly_d = poly_o.copy(); poly_d[:,0] = W - poly_d[:,0]
            polys_disp[mid] = poly_d
            cx, cy = int(poly_d[:,0].mean()), int(poly_d[:,1].mean())
            positions_by_id[mid] = (int(cx*(ANCHO/W)), int(cy*(ALTO/H)))

    for k in list(vis_count.keys()):
        if k not in current_ids: vis_count[k] = 0
    for k in current_ids:
        vis_count[k] = vis_count.get(k, 0) + 1
    visible_stable = {m for m,cnt in vis_count.items() if cnt >= VIS_CONSEC_REQ}

    t_ms = pygame.time.get_ticks()
    despawned = False

    # actualizar castor activo
    for mid, data in list(castores.items()):
        if mid in positions_by_id:
            data["pos"] = positions_by_id[mid]
            data["last_seen_ms"] = t_ms
            data["poly_orig"] = polys_orig.get(mid)
        # vida por tiempo
        if (t_ms - data["spawn_ms"]) >= TIEMPO_CASTOR_MS:
            vidas -= 1
            play_mp3_once(SND_PERDIDA)
            del castores[mid]; despawned = True; continue
        # puntos por tapado (desapareció + oclusión reciente)
        if mid not in current_ids:
            last_occ   = data.get("last_occluded_ms", -10**9)
            last_seen  = data.get("last_seen_ms",   -10**9)
            if (t_ms - last_occ) <= OCC_WINDOW_MS and (t_ms - last_seen) <= OCC_WINDOW_MS:
                ratio_now = motion_ratio_in_polygon(fgmask, data.get("poly_orig"))
                if ratio_now >= (MOTION_RATIO_HIT*0.7) or (t_ms - last_occ) <= 200:
                    puntos += 1
            del castores[mid]; despawned = True

    # registrar oclusión (solo si visible)
    for mid, data in list(castores.items()):
        if OCLUSION_ACTIVA and (mid in current_ids) and (data.get("poly_orig") is not None):
            if motion_ratio_in_polygon(fgmask, data["poly_orig"]) >= MOTION_RATIO_HIT:
                data["last_occluded_ms"] = t_ms

    # crear si no hay activo
    crear = (len(castores) == 0) and (not despawned)
    if crear and (t_ms - last_create_ms) >= INTERVALO_CREACION_MS:
        candidates = [m for m in MARCADORES.keys() if m in visible_stable]
        if candidates:
            mid = random.choice(candidates)
            castores[mid] = {
                "spawn_ms": t_ms,
                "pos": positions_by_id.get(mid, (ANCHO//2, ALTO//2)),
                "color": MARCADORES[mid],
                "last_seen_ms": t_ms,
                "last_occluded_ms": -10**9,
                "poly_orig": polys_orig.get(mid)
            }
            last_create_ms = t_ms

    # ---------- Dibujo ----------
    ventana.blit(pantalla_from_frame(frame_display), (0,0))
    for mid, info in castores.items():
        x,y = info["pos"]
        surf = castor_verde if info["color"]=="verde" else castor_morado
        ventana.blit(surf, (x - surf.get_width()//2, y - surf.get_height()//2))

    ventana.blit(fuente.render(f"Puntos: {puntos}", True, (0,255,0)), (20,20))

    if castores:
        cid, data = next(iter(castores.items()))
        remain_ms = max(0, TIEMPO_CASTOR_MS - (t_ms - data["spawn_ms"]))
        remain_sec = (remain_ms + 999)//1000
        timer_color = (255,255,255) if remain_sec>2 else ((255,165,0) if remain_sec>1 else (255,60,60))
        cron = fuente_timer.render(f"{remain_sec}s", True, timer_color)
        ventana.blit(cron, (ANCHO//2 - cron.get_width()//2, 8))

    dibujar_vidas(ventana, vidas)

    pygame.display.flip()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE: running = False
    clock.tick(30)

    # ---------- Game Over ----------
    if vidas <= 0:
        elapsed_s = (pygame.time.get_ticks() - start_time_ms)//1000
        hs = load_highscore()
        if puntos > hs:
            save_highscore(puntos); hs = puntos
        action = game_over_screen(puntos, elapsed_s, hs)
        if action == "restart":
            puntos, vidas = 0, VIDAS_MAX
            castores.clear(); vis_count.clear()
            last_create_ms = pygame.time.get_ticks()
            start_time_ms  = pygame.time.get_ticks()
            continue
        else:
            running = False
            break

# ---------- Salida ----------
cap.release()
pygame.quit()
cv2.destroyAllWindows()
sys.exit()
