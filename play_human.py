"""
play_human.py — Joue toi-même pour tester l'environnement

Contrôles :
    ↑ / Z   — Accélérer
    ↓ / S   — Freiner / Marche arrière
    ← / Q   — Virer à gauche
    → / D   — Virer à droite
    R       — Recommencer
    Échap   — Quitter
"""

import os
os.environ["SDL_VIDEODRIVER"] = "x11"
import pygame
import numpy as np
from car_env import CarRacingEnv

print("1. Initialisation de l'environnement...")
env  = CarRacingEnv(render_mode="human")
print("2. Reset de l'environnement...")
obs, _ = env.reset()
print("3. Rendu de l'environnement (ouverture fenêtre)...")
env.render()
print("4. Fenêtre ouverte avec succès !")
print(__doc__)

clock   = pygame.time.Clock()
running = True

while running:
    action = np.array([0.0, 0.0], dtype=np.float32)

    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                obs, _ = env.reset()
                print("↺  Restart")

    # Contrôles continus
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]  or keys[pygame.K_q]: action[0] = -1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] =  1.0
    if keys[pygame.K_UP]    or keys[pygame.K_z]: action[1] =  1.0
    if keys[pygame.K_DOWN]  or keys[pygame.K_s]: action[1] = -1.0

    obs, reward, done, _, info = env.step(action)
    env.render()

    if done:
        laps = info.get("laps", 0)
        print(f"💥 Crash ou fin d'épisode !  Tours : {laps}")
        obs, _ = env.reset()

    clock.tick(60)

env.close()
print("Au revoir !")
