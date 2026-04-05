"""
train.py — Entraînement PPO sur l'environnement CarRacingEnv

Usage :
    python train.py                  # 1 million de steps
    python train.py 2000000          # durée personnalisée
    python train.py 500000 models/ppo_car_500000_steps.zip  # reprendre depuis un checkpoint

Suivre la progression en temps réel :
    tensorboard --logdir logs/
"""

import sys
import os
os.environ["SDL_VIDEODRIVER"] = "x11"
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from car_env import CarRacingEnv

LOG_DIR   = "logs/"
MODEL_DIR = "models/"
EVAL_DIR  = "eval/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ─── Paramètres ──────────────────────────────────────────────────────────────
TOTAL_STEPS  = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
RESUME_FROM  = sys.argv[2] if len(sys.argv) > 2 else None

RENDER_TRAINING = True 
N_ENVS       = 12 # Équilibre parfait pour éviter les gels CPU

# ─── Création des envs ───────────────────────────────────────────────────────
from stable_baselines3.common.vec_env import DummyVecEnv

def make_custom_env(idx):
    def _init():
        # Gym exige que tous les envs aient le même render_mode
        render_mode = "human" if RENDER_TRAINING else None
        env = CarRacingEnv(render_mode=render_mode)
        env.env_idx = idx  # On ajoute cette variable pour identifier la voiture 0
        env.n_envs = N_ENVS # On transmet le nombre total pour synchro l'affichage
        return env
    return _init

train_env = DummyVecEnv([make_custom_env(i) for i in range(N_ENVS)])
# env unique pour évaluation (rendu activé pour éviter le freeze visuel)
render_mode = "human" if RENDER_TRAINING else None
eval_env  = Monitor(CarRacingEnv(render_mode=render_mode))

# ─── Modèle ──────────────────────────────────────────────────────────────────
if RESUME_FROM:
    print(f"➜  Reprise depuis {RESUME_FROM}...")
    model = PPO.load(RESUME_FROM, env=train_env, tensorboard_log=LOG_DIR)
else:
    print("➜  Création d'un nouveau modèle PPO...")
    model = PPO(
        "MlpPolicy",
        train_env,
        # ── Hyperparamètres PPO Optimisés ──
        learning_rate   = 1e-3,   
        n_steps         = 512,    # Moins de buffer = plus réactif et moins de gel
        batch_size      = 256,    # Batch réduit pour fluidifier les updates
        n_epochs        = 4,      # Divisé par 2.5 pour réduire le temps de freeze
        gamma           = 0.99,   
        gae_lambda      = 0.95,   
        clip_range      = 0.2,    
        ent_coef        = 0.01,   
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        policy_kwargs   = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
    )

# ─── Callbacks ───────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq   = 50_000 // N_ENVS,
    save_path   = MODEL_DIR,
    name_prefix = "ppo_car",
    verbose     = 1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = EVAL_DIR,
    log_path             = EVAL_DIR,
    eval_freq            = 20_000 // N_ENVS,
    n_eval_episodes      = 5,
    deterministic        = True,
    verbose              = 1,
)

# ─── Entraînement ────────────────────────────────────────────────────────────
print(f"\n🏁 Entraînement PPO — {TOTAL_STEPS:,} steps ({N_ENVS} envs parallèles)")
print("   Suivre en temps réel : tensorboard --logdir logs/\n")

model.learn(
    total_timesteps   = TOTAL_STEPS,
    callback          = [checkpoint_cb, eval_cb],
    progress_bar      = True,
    reset_num_timesteps = (RESUME_FROM is None),
)

final_path = f"{MODEL_DIR}ppo_car_final"
model.save(final_path)
print(f"\n✅ Entraînement terminé ! Modèle sauvegardé : {final_path}.zip")
print(f"   Meilleur modèle (eval) : {EVAL_DIR}best_model.zip")
print(f"\n   Pour le regarder jouer : python play.py")
