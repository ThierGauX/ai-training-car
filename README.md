# 🏎️ AI Car Racing — Reinforcement Learning

Une implémentation performante d'un environnement de course automobile 2D pour l'apprentissage par renforcement (**Reinforcement Learning**), utilisant **Stable-Baselines3** (PPO) et **Gymnasium**.

L'agent apprend à piloter une voiture sur un circuit complexe en utilisant uniquement des capteurs de distance (raycasts) et sa vitesse actuelle.

---

## ✨ Points Forts

*   🚀 **Entraînement Ultra-Rapide** : Optimisé pour tourner sur 12 environnements en parallèle (Synchro CPU/GPU).
*   🛰️ **Capteurs Raycast** : 7 capteurs de proximité simulant un Lidar (vision à 180°).
*   🎬 **Rendu Temps Réel** : Visualisez l'apprentissage en direct grâce à Pygame.
*   📊 **Monitoring complet** : Intégration TensorBoard pour suivre les récompenses et la perte (loss).
*   🛠️ **Physique Réaliste (Simplifiée)** : Gestion de l'inertie, de l'accélération et du freinage.

---

## 🛠️ Installation

```bash
# Cloner le dépôt (une fois créé)
# git clone https://github.com/votre-nom/ai_car.git
# cd ai_car

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation rapides

### 1. Mode Manuel (Jouer vous-même)
Testez la physique du véhicule et découvrez le tracé du circuit.
```bash
python play_human.py
```
*   **Z/S/Q/D** ou **Flèches** : Piloter
*   **R** : Recommencer

### 2. Entraînement de l'IA
Lancez l'apprentissage avec l'algorithme PPO.
```bash
python train.py 1000000          # Entraîner sur 1 million de steps
```
*Pendant l'entraînement, une phase d'évaluation automatique se lance tous les 20 000 steps pour enregistrer le meilleur modèle.*

### 3. Voir l'IA en action
Regardez votre agent entraîné piloter sur le circuit.
```bash
python play.py                   # Utilise le modèle final
```

---

## 🧠 Architecture Technique

| Composant | Description |
|-----------|-------------|
| **Observation** | 9 valeurs : 7 raycasts (0 à 250px), vitesse, angle vers le prochain checkpoint. |
| **Actions** | 2 valeurs continues [-1, 1] : Direction (Steer) et Accélération/Frein (Throttle). |
| **Algorithme** | **PPO** (Proximal Policy Optimization) - Stable-Baselines3. |
| **Environnement** | Custom Gymnasium Env (physique 2D Pygame). |

---

## 📊 Suivi des performances

Pour ouvrir le tableau de bord TensorBoard et voir les courbes d'apprentissage :
```bash
tensorboard --logdir logs/
```

---

## 🛤️ Personnalisation du circuit

Le circuit est défini par une série de points dans `car_env.py` (`CENTERLINE`). Vous pouvez créer vos propres tracés en modifiant ces coordonnées. L'environnement génèrera automatiquement les murs intérieurs et extérieurs en fonction de la largeur de piste souhaitée.

---

## 📄 Licence
Ce projet est sous licence MIT. Libre à vous de l'utiliser et de l'améliorer !
