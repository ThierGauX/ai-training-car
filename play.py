"""
play.py — Regarde l'agent entraîné conduire

Usage :
    python play.py                                   # charge models/ppo_car_final
    python play.py eval/best_model.zip               # meilleur modèle (EvalCallback)
    python play.py models/ppo_car_500000_steps.zip   # checkpoint intermédiaire
"""

import sys
import time
from stable_baselines3 import PPO
from car_env import CarRacingEnv

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_car_final"

print(f"Chargement du modèle : {model_path}")
env   = CarRacingEnv(render_mode="human")
model = PPO.load(model_path, env=env)

episode = 0
while True:
    obs, _    = env.reset()
    done      = False
    total_r   = 0.0
    episode  += 1
    print(f"\n── Épisode {episode} ──────────────────")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()
        total_r += reward

    laps = info.get("laps", 0)
    print(f"   Reward total : {total_r:.1f}  |  Tours complétés : {laps}")
    time.sleep(1)
