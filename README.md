# AI Car Racing - Reinforcement Learning

Une implementation performante d'un environnement de course automobile 2D pour l'apprentissage par renforcement (Reinforcement Learning), utilisant Stable-Baselines3 (PPO) et Gymnasium.

L'agent apprend a piloter une voiture sur un circuit complexe en utilisant uniquement des capteurs de distance (raycasts) et sa vitesse actuelle.

---

## Points Forts

*   Entrainement Ultra-Rapide : Optimise pour tourner sur 12 environnements en parallele (Synchro CPU/GPU).
*   Capteurs Raycast : 7 capteurs de proximite simulant un Lidar (vision a 180 degres).
*   Interface Multi-Pages : Navigation en temps reel entre la vue Circuit, la visualisation experte du reseau de neurones et le tableau de bord de telemetrie.
*   Monitoring complet : Integration TensorBoard pour suivre les recompenses et la perte (loss).
*   Physique Realiste (Simplifiee) : Gestion de l'inertie, de l'acceleration et du freinage.

---

## Installation

```bash
# Cloner le depot
git clone https://github.com/ThierGauX/ai-training-car.git
cd ai-training-car

# Installer les dependances
pip install -r requirements.txt
```

---

## Utilisation rapide

### 1. Mode Manuel (Jouer vous-meme)
Testez la physique du vehicule et decouvrez le trace du circuit.
```bash
python play_human.py
```
*   Z/S/Q/D ou Fleches : Piloter
*   R : Recommencer

### 2. Entrainement de l'IA
Lancez l'apprentissage avec l'algorithme PPO.
```bash
python train.py 1000000          # Entrainer sur 1 million de steps
```
Pendant l'entrainement, une phase d'evaluation automatique se lance tous les 20 000 steps pour enregistrer le meilleur modele.

### 3. Voir l'IA en action
Regardez votre agent entraine piloter sur le circuit en chargeant un point de controle ou le meilleur modele.
```bash
python play.py eval/best_model.zip
```

### 4. Interface Graphique Multi-Pages
Pendant l'execution de l'environnement (entraiment ou evaluation), vous disposez d'un menu superieur permettant de modifier radicalement l'affichage a l'aide de votre clavier :

*   Touche 1 (Circuit) : Affichage classique de la physionomie de la course.
*   Touche 2 (Cerveau IA) : Affichage plein ecran detaillant l'activation dynamique des noeuds du modele neuronal PPO selon les entrees lidars.
*   Touche 3 (Telemetrie) : Matrice statistique indiquant la recompense globale, le statut des episodes, et un suivi au pourcentage pres de la progression de l'entrainement.

---

## Architecture Technique

| Composant | Description |
|-----------|-------------|
| Observation | 9 valeurs : 7 raycasts (0 a 250px), vitesse, angle vers le prochain checkpoint. |
| Actions | 2 valeurs continues [-1, 1] : Direction (Steer) et Acceleration/Frein (Throttle). |
| Algorithme | PPO (Proximal Policy Optimization) - Stable-Baselines3. |
| Environnement | Custom Gymnasium Env (physique 2D Pygame). |

---

## Suivi des performances

Pour ouvrir le tableau de bord TensorBoard et analyser les courbes mathematiques :
```bash
tensorboard --logdir logs/
```

---

## Personnalisation du circuit

Le circuit est defini par une serie de points dans car_env.py (CENTERLINE). Vous pouvez generer des configurations differentes en alterant ces coordonnees. L'environnement regenerera le maillage global des bordures de piste a l'execution.

---

## Licence
Ce projet est sous licence MIT. Libre a vous de l'utiliser et de l'ameliorer.
