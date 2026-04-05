# 🏎 Car Racing RL

Environnement de course 2D entraîné avec **PPO** (Proximal Policy Optimization).
La voiture apprend à naviguer un circuit grâce à 7 capteurs raycast et une physique simplifiée.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Tester l'environnement à la main
```bash
python play_human.py
```
Contrôles : **↑↓←→** (ou ZQSD) pour conduire, **R** pour restart.

### 2. Lancer l'entraînement
```bash
python train.py                  # 1 million de steps (~20 min sur CPU)
python train.py 2000000          # durée personnalisée
```

Suivre la progression en temps réel :
```bash
tensorboard --logdir logs/
```

### 3. Regarder l'agent entraîné
```bash
python play.py                         # modèle final
python play.py eval/best_model.zip     # meilleur modèle (évaluation)
```

---

## Architecture

```
car_racing_rl/
├── car_env.py       ← Environnement Gymnasium (physique, raycasts, rendu)
├── train.py         ← Entraînement PPO (Stable-Baselines3)
├── play.py          ← Regarder l'agent IA jouer
├── play_human.py    ← Jouer soi-même au clavier
├── requirements.txt
├── models/          ← Checkpoints sauvegardés automatiquement
├── eval/            ← Meilleur modèle selon EvalCallback
└── logs/            ← Logs TensorBoard
```

---

## Observation (9 valeurs normalisées [0,1])

| Index | Description |
|-------|-------------|
| 0–6   | Distances aux murs (7 rayons : -90° à +90°) |
| 7     | Vitesse normalisée |
| 8     | Angle vers le prochain checkpoint (normalisé) |

## Actions (2 valeurs continues [-1, 1])

| Index | Description |
|-------|-------------|
| 0     | Steering : -1 = gauche, +1 = droite |
| 1     | Throttle : -1 = frein/marche arrière, +1 = accélérer |

## Récompenses

| Événement | Valeur |
|-----------|--------|
| Checkpoint franchi | +15 |
| Tour complet | +100 |
| Vitesse maintenue | +0.08 × speed / step |
| Pénalité temporelle | −0.02 / step |
| Crash (sortie de piste) | −50 + fin d'épisode |

---

## Personnalisation

**Changer le circuit** : modifie le tableau `CENTERLINE` dans `car_env.py`.  
Chaque point est `[x, y]` en pixels (écran 900×600, y vers le bas).  
Le circuit doit être **dans le sens horaire**.

**Rendre la voiture plus rapide** : augmente `MAX_SPEED` et `ACCELERATION`.

**Piste plus large** : augmente `TRACK_WIDTH` (défaut : 45px de chaque côté).

---

## Progression suggérée

1. ✅ Jouer soi-même pour comprendre l'env (`play_human.py`)
2. ✅ Lancer un premier entraînement court (200k steps) pour voir l'agent apprendre
3. 📈 Entraîner sur 1M steps et observer la courbe de reward sur TensorBoard
4. 🔬 Modifier les récompenses (reward shaping) et comparer
5. 🚗 Ajouter plusieurs voitures (multi-agent) avec `SubprocVecEnv`
6. 🗺 Générer des circuits aléatoires (curriculum learning)
