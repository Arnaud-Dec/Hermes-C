# Hermes-C 🚀

**Hermes-C** est un projet d'intelligence artificielle pour la prédiction financière (Bitcoin) basé sur **l'évolution génétique de réseaux de neurones**. Le projet est entièrement implémenté en **C pur** avec support **CPU multi-threadé** et **GPU (CUDA)** pour des performances optimales.

Contrairement aux approches traditionnelles qui utilisent la rétropropagation, Hermes-C utilise un algorithme génétique pour faire évoluer une population de réseaux de neurones, permettant une optimisation robuste sans calcul de gradients.

---

## 📂 Structure du Projet

* **`src/cpu/`** : Moteur d'évolution génétique multi-threadé (C standard).
* **`src/gpu/`** : Moteur d'évolution génétique accéléré GPU (CUDA).
* **`src/python/`** : Scripts pour télécharger les données historiques Bitcoin.
* **`include/`** : Headers partagés (NeuralNetwork.h, ThreadArgs.h).
* **`data/`** : Stockage des données brutes (`data.csv`).
* **`hermes_cpu`** / **`hermes_cuda`** : Exécutables compilés.

---

## ⚙️ Installation

### 1. Prérequis Python (Téléchargement des données)
Il est recommandé d'utiliser un environnement virtuel.

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation (Windows)
.\.venv\Scripts\activate

# Activation linux
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

### 2. Prérequis C/CUDA (Moteurs d'évolution)

#### Pour la version CPU :
* **Compilateur GCC** avec support pthread
* **Windows** : MinGW ou équivalent
* **Linux/Mac** : `sudo apt install build-essential` (ou Xcode)

#### Pour la version GPU (optionnelle) :
* **NVIDIA GPU** avec compute capability >= 3.0
* **CUDA Toolkit** installé (testé avec CUDA 11.0+)
* **Driver NVIDIA** compatible

#### Build Tools (optionnel) :
* **Make** : Pour utiliser les raccourcis d'automatisation du Makefile

---

## 🚀 Utilisation

Vous avez le choix entre utiliser les **raccourcis Makefile** ou compiler **manuellement**.

### 🌟 Méthode Rapide (Makefile)

#### Lancement CPU (Recommandé pour débuter) :
```bash
make full
```
*Télécharge les données → Compile la version CPU → Lance l'évolution*

#### Lancement GPU (Haute Performance) :
```bash
make data        # Télécharge les données
make cuda        # Compile la version GPU
./hermes_cuda    # Lance l'évolution sur GPU
```

#### Autres commandes utiles :
```bash
make cpu         # Compile uniquement la version CPU
make clean       # Supprime les exécutables
```

---

### 🛠️ Méthode Manuelle

#### Étape 1 : Récupération des Données 📉

Télécharge l'historique des prix du Bitcoin (BTC-USD) sur 2 ans via Yahoo Finance.

```bash
python src/python/get_data.py
```
> *Génère : `data/data.csv`*

#### Étape 2 : Compilation des Moteurs ⚙️

**Version CPU (Multi-thread) :**
```bash
gcc -Wall -Wextra -I include -pthread src/cpu/neural_evol.c -o hermes_cpu -lm -pthread
```

**Version GPU (CUDA) :**
```bash
nvcc -I include src/gpu/neural_evol.cu -o hermes_cuda
```

#### Étape 3 : Lancement de l'Évolution 🧬

**CPU :**
```bash
# Linux/Mac
./hermes_cpu

# Windows
hermes_cpu.exe
```

**GPU :**
```bash
# Linux/Mac
./hermes_cuda

# Windows
hermes_cuda.exe
```

Le programme va faire évoluer une population de réseaux de neurones pour prédire les mouvements de prix du Bitcoin.

---

## 🧠 Détails Techniques

### Algorithme Génétique

* **Population** : 1000 réseaux de neurones (individus)
* **Sélection** : Tournoi de fitness basé sur la précision des prédictions
* **Reproduction** : Croisement et mutation des poids/biais
* **Générations** : Évolution continue jusqu'à convergence optimale

### Architecture du Réseau (MLP)

* **Input Layer** : 60 neurones (Fenêtre glissante des 60 derniers rendements)
* **Hidden Layer** : 32 neurones (Activation ReLU)
* **Output Layer** : 1 neurone (Prédiction de variation, Activation Linéaire)

### Gestion des Données

* **Transformation** : Conversion prix → variations en pourcentage (× 10)
* **Calcul** : `((prix_jour - prix_hier) / prix_hier) * 10.0`
* **Fenêtre glissante** : 60 jours pour prédire le jour suivant

### Optimisations Performances

#### Version CPU :
* **Multi-threading** : Parallélisation de l'évaluation des individus
* **Pthread** : Distribution de la charge sur tous les cœurs disponibles

#### Version GPU :
* **CUDA** : Calculs massivement parallèles sur GPU NVIDIA
* **Accélération** : ~10-50x plus rapide selon la carte graphique

### Fitness et Évaluation

La fitness de chaque réseau est calculée sur l'erreur absolue entre la prédiction et la valeur réelle de variation (`fabs(prediction - target)`). Plus l'erreur est faible, meilleure est la fitness.

## 🚀 Fonctionnalités

### 💡 Innovation
* **Pas de rétropropagation** : L'évolution génétique remplace le calcul de gradients
* **Pur C/CUDA** : Performances maximales sans dépendances lourdes (Python/PyTorch)
* **Évolution en temps réel** : Visualisation de l'amélioration des prédictions génération par génération

### ⚡ Performance
* **Multi-plateforme** : Windows, Linux, macOS
* **Scalabilité** : Du CPU simple cœur au GPU haute performance
* **Efficacité mémoire** : Optimisé pour fonctionner même sur des systèmes embarqués

### 🎯 Application
* **Trading algorithmique** : Base pour des stratégies de trading automatisées
* **Recherche IA** : Démonstration d'alternatives à l'apprentissage supervisé traditionnel
* **Éducation** : Code source complet pour comprendre les algorithmes génétiques appliqués au ML

---

## 📜 Licence
Projet open-source sous licence **MIT**.

## Version
**v2.0.0-ALPHA** - Évolution Génétique