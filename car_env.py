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
SCREEN_W, SCREEN_H = 900, 600
FPS          = 60
TRACK_WIDTH  = 45      # demi-largeur de la piste (pixels)
MAX_RAY_LEN  = 250.0   # portée max des capteurs
MAX_SPEED    = 12.0
ACCELERATION = 0.45
FRICTION     = 0.06
TURN_SPEED   = 0.065   # rad/step

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
class CarRacingEnv(gym.Env):
    """
    Environnement de course 2D avec :
      - 7 capteurs raycast (distances aux murs)
      - Physique voiture simplifiée
      - Récompense : checkpoints + vitesse − crash

    Observation (9 valeurs, normalisées [0,1]) :
        [ray0..ray6, vitesse, angle_vers_prochain_checkpoint]

    Actions continues (2 valeurs dans [-1,1]) :
        [steering, throttle]
    """
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.zeros(9, dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2,  1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.screen      = None
        self.clock       = None
        self.font        = None
        self._pg_ready   = False

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _get_obs(self):
        rays      = cast_rays(self._pos, self._heading) / MAX_RAY_LEN

        cp        = CHECKPOINTS[self._next_cp]
        target_a  = math.atan2(cp[1] - self._pos[1], cp[0] - self._pos[0])
        diff      = (target_a - self._heading + math.pi) % (2 * math.pi) - math.pi
        angle_n   = np.clip(diff / math.pi * 0.5 + 0.5, 0, 1)

        speed_n   = np.clip(
            (self._speed + MAX_SPEED * 0.3) / (MAX_SPEED * 1.3), 0, 1
        )
        return np.append(rays, [float(speed_n), float(angle_n)]).astype(np.float32)

    # ── Gym API ───────────────────────────────────────────────────────────────
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
        return self._get_obs(), {}

    def step(self, action):
        steer    = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], -1, 1))

        # Physique
        sensitivity       = 0.3 + abs(self._speed) / MAX_SPEED
        self._heading    += steer * TURN_SPEED * sensitivity
        self._speed      += throttle * ACCELERATION
        self._speed      -= np.sign(self._speed) * FRICTION
        self._speed       = float(np.clip(self._speed, -MAX_SPEED * 0.3, MAX_SPEED))

        self._pos += np.array(
            [math.cos(self._heading) * self._speed,
             math.sin(self._heading) * self._speed],
            dtype=np.float32,
        )
        self._steps += 1

        # Crash → grosse pénalité, fin d'épisode
        if not point_in_track(self._pos):
            return self._get_obs(), -50.0, True, False, {"laps": self._laps}

        reward = 0.0

        # Checkpoint franchi
        if np.linalg.norm(self._pos - CHECKPOINTS[self._next_cp]) < TRACK_WIDTH * 1.5:
            reward += 15.0
            was_last        = (self._next_cp == N_CP - 1)
            self._next_cp   = (self._next_cp + 1) % N_CP
            if was_last:
                self._laps += 1
                reward     += 150.0   # bonus tour complet ++

        reward += max(self._speed, 0) * 0.25    # GROS bonus vitesse (0.08 -> 0.25)
        reward -= 0.05                          # pénalité temporelle accrue

        self._total_r += reward
        done = self._steps >= 3000

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, done, False, {"laps": self._laps}

    # ── Rendu ─────────────────────────────────────────────────────────────────
    def render(self):
        if not self._pg_ready:
            pygame.init()
            self.screen  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("🏎  Car Racing RL")
            self.clock   = pygame.time.Clock()
            self.font    = pygame.font.SysFont("monospace", 15)
            self._pg_ready = True

        env_idx = getattr(self, "env_idx", 0)
        n_envs  = getattr(self, "n_envs", 1)

        # La voiture 0 s'occupe d'effacer l'écran et dessiner le décor
        if env_idx == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close(); return

            self.screen.fill((18, 18, 18))

            # — Piste —
            pygame.draw.polygon(self.screen, (52, 52, 52), OUTER_WALL.astype(int).tolist())
            pygame.draw.polygon(self.screen, (18, 18, 18), INNER_WALL.astype(int).tolist())

            # — Murs —
            pygame.draw.polygon(self.screen, (200, 200, 200), OUTER_WALL.astype(int).tolist(), 3)
            pygame.draw.polygon(self.screen, (200, 200, 200), INNER_WALL.astype(int).tolist(), 3)

            # — Ligne de départ (jaune) —
            s   = CENTERLINE[0]
            d0  = CENTERLINE[1] - CENTERLINE[0]
            d0 /= np.linalg.norm(d0)
            p   = _perp_ccw(d0)
            pygame.draw.line(
                self.screen, (255, 220, 0),
                (s + p * TRACK_WIDTH).astype(int),
                (s - p * TRACK_WIDTH).astype(int), 3,
            )

            # — Prochain checkpoint (vert) pour la voiture 0 —
            cp = CHECKPOINTS[self._next_cp]
            pygame.draw.circle(self.screen, (0, 210, 90), cp.astype(int), 8, 2)

            # — Raycasts (seulement pour la voiture 0 pour ne pas surcharger) —
            rays = cast_rays(self._pos, self._heading)
            for i, rel_a in enumerate(RAY_ANGLES_REL):
                angle = self._heading + rel_a
                end   = self._pos + np.array([math.cos(angle), math.sin(angle)]) * rays[i]
                col   = (220, 60, 30) if rays[i] < 60 else (55, 80, 190)
                pygame.draw.line(self.screen, col, self._pos.astype(int), end.astype(int), 1)

            # — HUD —
            mode_text = "EVALUATION" if n_envs == 1 else "ENTRAÎNEMENT"
            hud = (f"[{mode_text}] Vitesse: {self._speed:+.1f}  |  "
                   f"Tours: {self._laps}  |  "
                   f"Voitures actives: {n_envs}")
            color = (255, 200, 50) if n_envs == 1 else (190, 190, 190)
            self.screen.blit(self.font.render(hud, True, color), (8, 8))

        # — Toutes les voitures se dessinent elles-mêmes —
        ch, sh = math.cos(self._heading), math.sin(self._heading)
        front = self._pos + np.array([ch,  sh]) * 13
        left  = self._pos + np.array([-sh,  ch]) * 6 - np.array([ch, sh]) * 7
        right = self._pos + np.array([ sh, -ch]) * 6 - np.array([ch, sh]) * 7
        
        # La voiture 0 est rouge, les autres sont bleues transparentes
        color = (235, 65, 65) if env_idx == 0 else (65, 150, 235)
        pygame.draw.polygon(self.screen, color,
                            [front.astype(int), left.astype(int), right.astype(int)])

        # La dernière voiture demande la mise à jour de l'écran visible !
        if env_idx == n_envs - 1:
            pygame.display.flip()
            # self.clock.tick(FPS) # <--- Désactivé pour accélérer le temps au maximum !

    def close(self):
        if self._pg_ready:
            pygame.quit()
            self._pg_ready = False
