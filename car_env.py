"""
car_env.py — Environnement de course 2D pour Reinforcement Learning
Compatible Gymnasium + Stable-Baselines3 (PPO)
"""

import math
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

# ─── Configuration ───────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1200, 600
TRACK_W_UI   = 900     # Largeur de la zone de piste
SIDEBAR_W    = 300     # Largeur de la barre latérale
FPS          = 60
TRACK_WIDTH  = 45      # demi-largeur de la piste (pixels)
MAX_RAY_LEN  = 250.0   # portée max des capteurs
MAX_SPEED    = 12.0
ACCELERATION = 0.45
FRICTION     = 0.06
TURN_SPEED   = 0.065   # rad/step

# ── Couleurs Premium ──
C_BG         = (10, 10, 12)
C_SIDEBAR    = (22, 22, 26)
C_TRACK      = (35, 35, 40)
C_WALL       = (80, 80, 95)
C_ACCENT     = (0, 200, 255) # Cyan néon
C_ACCENT_2   = (255, 40, 100) # Rose néon
C_TEXT       = (230, 230, 240)
C_GAUGE_BG   = (40, 40, 50)

# ─── Tracé du circuit ────────────────────────────────────────────────────────
# Points du centre de la piste (sens horaire en coords écran)
# Tu peux modifier ces valeurs pour changer la forme du circuit !
CENTERLINE = np.array([
    [150, 430],   # 0  Départ / Arrivée
    [150, 160],   # 1  Montée gauche
    [240,  90],   # 2  Virage haut-gauche
    [430,  70],   # 3  Ligne droite haute
    [540,  70],   # 4  Entrée chicane
    [580, 135],   # 5  Chicane (bosse vers le bas)
    [640,  70],   # 6  Sortie chicane
    [760,  70],   # 7  Ligne droite haute-droite
    [840, 160],   # 8  Virage haut-droit
    [840, 430],   # 9  Descente droite
    [750, 510],   # 10 Virage bas-droit
    [450, 510],   # 11 Ligne droite basse
    [240, 510],   # 12 Virage bas-gauche
], dtype=float)

N_CP = len(CENTERLINE)

# ─── Calcul automatique des murs ─────────────────────────────────────────────
def _perp_ccw(v):
    """Rotation 90° dans le sens trigo d'un vecteur 2D."""
    return np.array([-v[1], v[0]])

def _compute_walls(cl, width):
    """
    Pour un tracé horaire :
      perp_ccw → pointe vers l'intérieur du circuit (inner wall / îlot)
      -perp_ccw → pointe vers l'extérieur (outer wall / barrière)
    """
    n = len(cl)
    inner_pts, outer_pts = [], []
    for i in range(n):
        prev_pt = cl[(i - 1) % n]
        curr_pt = cl[i]
        next_pt = cl[(i + 1) % n]

        d1 = curr_pt - prev_pt;  d1 /= np.linalg.norm(d1) + 1e-10
        d2 = next_pt - curr_pt;  d2 /= np.linalg.norm(d2) + 1e-10

        d_avg = d1 + d2
        nm = np.linalg.norm(d_avg)
        d_avg = d_avg / nm if nm > 1e-10 else d1

        perp = _perp_ccw(d_avg)
        inner_pts.append(curr_pt + perp * width)   # intérieur
        outer_pts.append(curr_pt - perp * width)   # extérieur
    return np.array(inner_pts), np.array(outer_pts)

INNER_WALL, OUTER_WALL = _compute_walls(CENTERLINE, TRACK_WIDTH)

# Checkpoints : milieux des segments du tracé central
CHECKPOINTS = np.array([
    (CENTERLINE[i] + CENTERLINE[(i + 1) % N_CP]) / 2
    for i in range(N_CP)
])

# ─── Murs → segments pour raycast ────────────────────────────────────────────
def _poly_to_segs(poly):
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]

WALL_SEGS = _poly_to_segs(INNER_WALL) + _poly_to_segs(OUTER_WALL)

# ─── Raycast ─────────────────────────────────────────────────────────────────
RAY_ANGLES_REL = np.deg2rad([-90, -45, -20, 0, 20, 45, 90])   # 7 capteurs

def _ray_seg_dist(orig, d, a, b):
    """Distance origine→intersection rayon/segment. np.inf si aucune."""
    v = b - a
    cross = d[0] * v[1] - d[1] * v[0]
    if abs(cross) < 1e-10:
        return np.inf
    w = a - orig
    t = (w[0] * v[1] - w[1] * v[0]) / cross
    u = (w[0] * d[1] - w[1] * d[0]) / cross
    return t if (t > 0 and 0 <= u <= 1) else np.inf

def cast_rays(pos, heading):
    """Retourne les 7 distances aux murs (non normalisées)."""
    dists = []
    for rel_a in RAY_ANGLES_REL:
        angle = heading + rel_a
        direc = np.array([math.cos(angle), math.sin(angle)])
        min_d = MAX_RAY_LEN
        for a, b in WALL_SEGS:
            d = _ray_seg_dist(pos, direc, a, b)
            if d < min_d:
                min_d = d
        dists.append(min_d)
    return np.array(dists, dtype=np.float32)

# ─── Détection piste ─────────────────────────────────────────────────────────
def _pip(px, py, poly):
    """Point-in-polygon par lancer de rayon."""
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i];  xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside

def point_in_track(pos):
    x, y = float(pos[0]), float(pos[1])
    return _pip(x, y, OUTER_WALL) and not _pip(x, y, INNER_WALL)


# ─── Environnement Gymnasium ──────────────────────────────────────────────────
import multiprocessing as mp

# ─── Moteur de Rendu Multiprocessus (Anti-Freeze) ───────────────────────────
class RenderProcess(mp.Process):
    """Processus dédié au rendu Pygame pour éviter les freezes système."""
    def __init__(self, queue, cmd_queue):
        super().__init__()
        self.queue = queue
        self.cmd_queue = cmd_queue

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🏎  AI Car Racing — DeepMind Training")
        clock = pygame.time.Clock()
        
        try:
            font = pygame.font.SysFont("Segoe UI", 16, bold=True)
            font_main = pygame.font.SysFont("Segoe UI", 24, bold=True)
            font_dash = pygame.font.SysFont("Consolas", 14)
        except:
            font = pygame.font.SysFont("monospace", 15)
            font_main = pygame.font.SysFont("monospace", 20)
            font_dash = pygame.font.SysFont("monospace", 13)

        running = True
        last_states = {}
        n_envs = 1
        agent_status = "GATHERING EXP..."
        sim_delay = 0.0
        paused = False
        current_page = 0
        progress_pct = 0.0

        while running:
            # On vide la queue pour avoir le dernier état de chaque voiture
            while not self.queue.empty():
                try:
                    msg = self.queue.get_nowait()
                    if msg == "QUIT": running = False; break
                    if isinstance(msg, tuple):
                        if msg[0] == "STATUS": agent_status = msg[1]
                        elif msg[0] == "PROGRESS": progress_pct = float(msg[1])
                        else:
                            idx, state, n_e = msg
                            last_states[idx] = state
                            n_envs = n_e
                    else:
                        pass
                except: break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        self.cmd_queue.put("TOGGLE_PAUSE")
                    elif event.key == pygame.K_UP:
                        sim_delay = max(0.0, sim_delay - 0.005)
                        self.cmd_queue.put("SPEED_UP")
                    elif event.key == pygame.K_DOWN:
                        sim_delay += 0.005
                        self.cmd_queue.put("SPEED_DOWN")
                    elif event.key == pygame.K_1: current_page = 0
                    elif event.key == pygame.K_2: current_page = 1
                    elif event.key == pygame.K_3: current_page = 2

            if not running: break

            # Rendu
            screen.fill(C_BG)
            
            # Vues
            if current_page == 0:
                pygame.draw.rect(screen, C_SIDEBAR, (0, 0, TRACK_W_UI, SCREEN_H))
                pygame.draw.polygon(screen, C_TRACK, OUTER_WALL.astype(int).tolist())
                pygame.draw.polygon(screen, C_BG, INNER_WALL.astype(int).tolist())
                pygame.draw.polygon(screen, C_WALL, OUTER_WALL.astype(int).tolist(), 2)
                pygame.draw.polygon(screen, C_WALL, INNER_WALL.astype(int).tolist(), 2)

                s, d0 = CENTERLINE[0], CENTERLINE[1] - CENTERLINE[0]
                d0 /= np.linalg.norm(d0); p = _perp_ccw(d0)
                pygame.draw.line(screen, (255, 215, 0), (s + p * TRACK_WIDTH).astype(int), (s - p * TRACK_WIDTH).astype(int), 4)

                for idx, s in last_states.items():
                    pos, head, speed, rays, steps, total_r, laps, next_cp, obs, action = s
                    ch, sh = math.cos(head), math.sin(head)
                    f = pos + np.array([ch, sh]) * 13
                    l = pos + np.array([-sh, ch]) * 6 - np.array([ch, sh]) * 7
                    r = pos + np.array([sh, -ch]) * 6 - np.array([ch, sh]) * 7
                    
                    color = C_ACCENT_2 if idx == 0 else (60, 100, 150)
                    pygame.draw.polygon(screen, color, [f.astype(int), l.astype(int), r.astype(int)])

                    if idx == 0:
                        for i, rel_a in enumerate(RAY_ANGLES_REL):
                            angle = head + rel_a
                            end = pos + np.array([math.cos(angle), math.sin(angle)]) * rays[i]
                            pygame.draw.line(screen, (C_ACCENT if rays[i] > 60 else C_ACCENT_2), pos.astype(int), end.astype(int), 1)
                        
                        self._draw_sidebar(screen, font_main, font, font_dash, n_envs, s, agent_status, sim_delay, paused)

            elif current_page == 1 and 0 in last_states:
                self._draw_neural_net_page(screen, font_main, font, font_dash, last_states[0])

            elif current_page == 2 and 0 in last_states:
                self._draw_dashboard_page(screen, font_main, font, font_dash, last_states[0], progress_pct, agent_status, n_envs)

            # TOP NAV BAR
            nav_bg = pygame.Surface((SCREEN_W, 40), pygame.SRCALPHA)
            nav_bg.fill((22, 22, 26, 230))
            screen.blit(nav_bg, (0,0))
            
            nav_texts = ["[1] CIRCUIT", "[2] CERVEAU IA", "[3] TÉLÉMÉTRIE"]
            for i, txt in enumerate(nav_texts):
                color = C_ACCENT if current_page == i else (120, 120, 130)
                surf = font.render(txt, True, color)
                screen.blit(surf, (SCREEN_W//2 - 250 + i*200, 10))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def _draw_sidebar(self, screen, f_main, f_std, f_dash, n_envs, state, agent_status, sim_delay, paused):
        pos, head, speed, rays, steps, total_r, laps, next_cp, obs, action = state
        pygame.draw.rect(screen, C_SIDEBAR, (TRACK_W_UI, 0, SIDEBAR_W, SCREEN_H))
        pygame.draw.line(screen, C_WALL, (TRACK_W_UI, 0), (TRACK_W_UI, SCREEN_H), 2)

        x, y = TRACK_W_UI + 20, 30
        screen.blit(f_main.render("MISSION CONTROL", True, C_ACCENT), (x, y)); y+=50
        
        mode = "EVALUATION" if n_envs == 1 else "TRAINING"
        col = (255,180,0) if n_envs == 1 else C_ACCENT
        screen.blit(f_std.render(f"MODE: {mode}", True, col), (x, y)); y+=35

        # Status
        screen.blit(f_dash.render(f"PPO: {agent_status}", True, C_ACCENT_2), (x, y)); y+=35
        spd = f"{(1.0 / sim_delay):.0f} FPS Target" if sim_delay > 0 else "UNLIMITED (Fastest)"
        if paused: spd = "PAUSED"
        screen.blit(f_dash.render(f"SIM SPEED: {spd}", True, (150,255,150)), (x, y)); y+=40

        def gauge(label, val, m, yp, c):
            screen.blit(f_dash.render(label, True, C_TEXT), (x, yp))
            pygame.draw.rect(screen, C_GAUGE_BG, (x, yp+20, 260, 12))
            fw = int(np.clip(val/m, 0, 1) * 260)
            pygame.draw.rect(screen, c, (x, yp+20, fw, 12))
            screen.blit(f_dash.render(f"{val:.1f}", True, C_TEXT), (x+220, yp))

        gauge("VEHICLE SPEED", speed, MAX_SPEED, y, C_ACCENT); y+=55
        gauge("PROGRESSION", steps, 3000, y, (150,150,150)); y+=55

        stats = [("TOTAL LAPS", str(laps)), ("REWARD", f"{total_r:.1f}"), ("ENVS ACTIVE", str(n_envs))]
        for l, v in stats:
            screen.blit(f_dash.render(l, True, (120,120,130)), (x, y))
            screen.blit(f_std.render(v, True, C_TEXT), (x+140, y)); y+=35

    def _draw_neural_net_page(self, screen, f_main, f_std, f_dash, state):
        pos, head, speed, rays, steps, total_r, laps, next_cp, obs, action = state
        
        screen.blit(f_main.render("ARCHITECTURE DU CERVEAU IA (PPO)", True, C_ACCENT), (50, 60))
        
        labels_in = ["Laser Gauche 90", "Laser Gauche 45", "Laser Gauche 20", "Laser Face 0", "Laser Droit 20", "Laser Droit 45", "Laser Droit 90", "Vitesse Act.", "Angle Cible"]
        labels_out = ["Volant (G/D)", "Pédales (Frein/Acc.)"]
        
        layers = [9, 8, 8, 2] # 4 layers
        nn_w = 700
        nn_h = 450
        start_x = 250
        start_y = 100
        
        layer_x = [start_x + i * (nn_w / 3) for i in range(4)]
        nodes = []
        for l_idx, count in enumerate(layers):
            cx = layer_x[l_idx]
            dy = nn_h / (count + 1)
            layer_nodes = []
            for i in range(count):
                layer_nodes.append((cx, start_y + dy * (i + 1)))
            nodes.append(layer_nodes)
            
        for l in range(len(nodes) - 1):
            for i, p1 in enumerate(nodes[l]):
                for j, p2 in enumerate(nodes[l+1]):
                    if l == 0: val = obs[i]
                    elif l == len(nodes) - 2: val = (action[j] + 1) / 2
                    else: val = (obs[i % len(obs)] + (action[j % len(action)] + 1)/2) / 2
                    val = max(0.0, min(1.0, float(val)))
                    
                    alpha = 0.1 + 0.9 * val
                    r = int(C_BG[0] * (1-alpha) + 40 * alpha)
                    g = int(C_BG[1] * (1-alpha) + 200 * alpha)
                    b = int(C_BG[2] * (1-alpha) + 255 * alpha)
                    pygame.draw.line(screen, (r, g, b), p1, p2, 1 if val < 0.5 else 2)

        for l, l_nodes in enumerate(nodes):
            for i, p in enumerate(l_nodes):
                if l == 0:
                    val = obs[i]
                    c_base = C_ACCENT if i < 7 else C_ACCENT_2
                    rad = 6 + int(6 * val)
                    label = f_std.render(labels_in[i], True, (150,150,160))
                    screen.blit(label, (p[0] - label.get_width() - 25, p[1] - label.get_height()//2))
                elif l == 3:
                    val = (action[i] + 1) / 2
                    c_base = (255, 100, 50) if action[i] < 0 else (50, 255, 100)
                    rad = 8 + int(8 * abs(action[i]))
                    label = f_std.render(labels_out[i], True, (200,200,210))
                    screen.blit(label, (p[0] + 25, p[1] - label.get_height()//2))
                    
                    sub = f_dash.render(f"{action[i]:.2f}", True, c_base)
                    screen.blit(sub, (p[0] + 25, p[1] + 10))
                else: 
                    val = (obs[i % len(obs)] + (action[i % len(action)] + 1)/2) / 2
                    c_base = (100, 150, 200)
                    rad = 4 + int(5 * val)
                
                intensity = 100 + 155 * float(val)
                c_node = (min(255, int(c_base[0]*intensity/255)), 
                          min(255, int(c_base[1]*intensity/255)), 
                          min(255, int(c_base[2]*intensity/255)))
                pygame.draw.circle(screen, c_node, (int(p[0]), int(p[1])), rad)

    def _draw_dashboard_page(self, screen, f_main, f_std, f_dash, state, progress_pct, agent_status, n_envs):
        pos, head, speed, rays, steps, total_r, laps, next_cp, obs, action = state
        
        screen.blit(f_main.render("TÉLÉMÉTRIE & STATISTIQUES GLOBALES", True, C_ACCENT), (50, 60))
        
        y = 150
        screen.blit(f_std.render("PROGRESSION DE L'ENTRAÎNEMENT GLOBAL", True, (200,200,220)), (50, y))
        y += 30
        pygame.draw.rect(screen, C_GAUGE_BG, (50, y, 1100, 40))
        fw = int(progress_pct * 1100)
        pygame.draw.rect(screen, C_ACCENT, (50, y, fw, 40))
        pct_txt = f_main.render(f"{progress_pct*100:.2f}%", True, C_BG)
        screen.blit(pct_txt, (50 + fw // 2 - 30 if fw > 60 else 60, y + 5))
        
        y += 100
        boxes = [
            ("STATUT IA", agent_status),
            ("RECOMPENSE EPISODE", f"{total_r:.1f}"),
            ("NOMBRES DE TOURS", f"{laps}"),
            ("VITESSE VEHICULE", f"{speed:.1f} / {MAX_SPEED:.1f}"),
            ("ENVS (COEURS) ACTIFS", f"{n_envs}"),
            ("STEPS (EPISODE)", f"{steps} / 3000")
        ]
        
        try: font_big = pygame.font.SysFont("Segoe UI", 32, bold=True)
        except: font_big = pygame.font.SysFont("monospace", 28, bold=True)

        for i, (label, val) in enumerate(boxes):
            bx = 50 + (i % 3) * 380
            by = y + (i // 3) * 120
            pygame.draw.rect(screen, C_SIDEBAR, (bx, by, 340, 100))
            pygame.draw.rect(screen, C_WALL, (bx, by, 340, 100), 2)
            
            screen.blit(f_dash.render(label, True, (150,150,160)), (bx + 20, by + 20))
            screen.blit(font_big.render(val, True, C_TEXT), (bx + 20, by + 45))

# ─── Environnement Gymnasium ──────────────────────────────────────────────────
class CarRacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}
    _ui_queue = mp.Queue(maxsize=20)
    _cmd_queue = mp.Queue(maxsize=100)
    _ui_proc  = None
    _sim_delay = 0.0
    _paused = False

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        if render_mode == "human" and CarRacingEnv._ui_proc is None:
            CarRacingEnv._ui_proc = RenderProcess(CarRacingEnv._ui_queue, CarRacingEnv._cmd_queue)
            CarRacingEnv._ui_proc.start()

    def _get_obs(self):
        rays      = cast_rays(self._pos, self._heading) / MAX_RAY_LEN
        cp        = CHECKPOINTS[self._next_cp]
        target_a  = math.atan2(cp[1] - self._pos[1], cp[0] - self._pos[0])
        diff      = (target_a - self._heading + math.pi) % (2 * math.pi) - math.pi
        angle_n   = np.clip(diff / math.pi * 0.5 + 0.5, 0, 1)
        speed_n   = np.clip((self._speed + MAX_SPEED * 0.3) / (MAX_SPEED * 1.3), 0, 1)
        obs = np.append(rays, [float(speed_n), float(angle_n)]).astype(np.float32)
        self._last_obs = obs
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos     = CENTERLINE[0].copy().astype(np.float32)
        d0            = CENTERLINE[1] - CENTERLINE[0]
        self._heading = float(math.atan2(d0[1], d0[0]))
        self._speed   = 0.0
        self._next_cp = 1
        self._steps   = 0
        self._laps    = 0
        self._total_r = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        import time
        if getattr(self, "env_idx", 0) == 0:
            while not CarRacingEnv._cmd_queue.empty():
                try:
                    cmd = CarRacingEnv._cmd_queue.get_nowait()
                    if cmd == "TOGGLE_PAUSE": CarRacingEnv._paused = not CarRacingEnv._paused
                    elif cmd == "SPEED_UP": CarRacingEnv._sim_delay = max(0.0, CarRacingEnv._sim_delay - 0.005)
                    elif cmd == "SPEED_DOWN": CarRacingEnv._sim_delay += 0.005
                except: pass

        while CarRacingEnv._paused:
            time.sleep(0.1)
            if getattr(self, "env_idx", 0) == 0:
                while not CarRacingEnv._cmd_queue.empty():
                    try:
                        cmd = CarRacingEnv._cmd_queue.get_nowait()
                        if cmd == "TOGGLE_PAUSE": CarRacingEnv._paused = False
                    except: pass

        if CarRacingEnv._sim_delay > 0:
            time.sleep(CarRacingEnv._sim_delay)

        self._last_action = action
        steer, throttle = float(action[0]), float(action[1])
        sensitivity       = 0.3 + abs(self._speed) / MAX_SPEED
        self._heading    += steer * TURN_SPEED * sensitivity
        self._speed      += throttle * ACCELERATION
        self._speed      -= np.sign(self._speed) * FRICTION
        self._speed       = float(np.clip(self._speed, -MAX_SPEED * 0.3, MAX_SPEED))
        self._pos += np.array([math.cos(self._heading) * self._speed, math.sin(self._heading) * self._speed], dtype=np.float32)
        self._steps += 1

        if not point_in_track(self._pos):
            return self._get_obs(), -50.0, True, False, {"laps": self._laps}

        reward = 0.0
        if np.linalg.norm(self._pos - CHECKPOINTS[self._next_cp]) < TRACK_WIDTH * 1.5:
            reward += 15.0
            was_last = (self._next_cp == N_CP - 1)
            self._next_cp = (self._next_cp + 1) % N_CP
            if was_last: self._laps += 1; reward += 150.0

        reward += max(self._speed, 0) * 0.25
        reward -= 0.05
        self._total_r += reward
        done = self._steps >= 3000

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, done, False, {"laps": self._laps}

    def render(self):
        env_idx = getattr(self, "env_idx", 0)
        n_envs  = getattr(self, "n_envs", 1)
        state = (self._pos.copy(), self._heading, self._speed, cast_rays(self._pos, self._heading), self._steps, self._total_r, self._laps, self._next_cp, self._last_obs, getattr(self, "_last_action", np.zeros(2, dtype=np.float32)))
        try:
            self._ui_queue.put_nowait((env_idx, state, n_envs))
        except: pass

    def close(self):
        if CarRacingEnv._ui_proc:
            self._ui_queue.put("QUIT")
            CarRacingEnv._ui_proc.join()
            CarRacingEnv._ui_proc = None
