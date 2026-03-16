# Hermes-C 🚀

**Hermes-C** est un projet hybride d'intelligence artificielle pour la prédiction financière (Bitcoin), conçu pour démontrer l'implémentation d'un moteur d'inférence en **C Pur**.

Le projet sépare l'entraînement (Python/PyTorch) de l'exécution (C), permettant de déployer des modèles de Deep Learning sur des systèmes embarqués ou haute performance sans dépendance lourde (pas de Python ni de PyTorch requis pour l'exécution).

---

## 📂 Structure du Projet

* **`src/python/`** : Scripts pour télécharger les données et entraîner le modèle.
* **`src/c/`** : Moteur d'inférence en C (lit le binaire et le CSV).
* **`data/`** : Stockage des données brutes (`data.csv`).
* **`models/`** : Stockage du modèle binaire exporté (`model.bin`).

---

## ⚙️ Installation

### 1. Prérequis Python (Entraînement)
Il est recommandé d'utiliser un environnement virtuel.

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation (Windows)
.\.venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

### 2. Prérequis C (Moteur)

Un compilateur GCC est nécessaire.

* **Windows** : MinGW ou équivalent.
* **Linux/Mac** : `sudo apt install build-essential` (ou Xcode).
* *(Optionnel)* **Make** : Pour utiliser les raccourcis d'automatisation.

---

## 🚀 Utilisation

Vous avez le choix entre exécuter le projet **manuellement** étape par étape, ou utiliser les raccourcis du **Makefile** si l'outil est installé sur votre machine.

### 🌟 Méthode Rapide (Tout-en-un avec Make)

Si vous souhaitez tout lancer en une seule commande (Téléchargement -> Entraînement -> Compilation -> Prédiction) :

```bash
make full

```

---

### 🛠️ Méthode Étape par Étape

#### Étape 1 : Récupération des Données 📉

Télécharge l'historique des prix du Bitcoin (BTC-USD) sur 2 ans via Yahoo Finance.

* **Manuellement :**
```bash
python src/python/get_data.py

```


* **Via Makefile :**
```bash
make data

```

> *Génère : `data/data.csv*`

#### Étape 2 : Entraînement du Modèle 🧠

Entraîne le réseau de neurones avec PyTorch. Le script normalise les données, entraîne le modèle, et **exporte les poids et la configuration** dans un fichier binaire optimisé pour le C.

* **Manuellement :**
```bash
python src/python/train_model.py

```

* **Via Makefile :**
```bash
make train

```

> *Génère : `models/model.bin` (Contient Min, Max, Poids et Biais)*

#### Étape 3 : Compilation et Prédiction (Moteur C) ⚡

Le moteur C charge le fichier binaire et les données CSV pour prédire le prix de demain.

**1. Compilation :**

* **Manuellement :**
```bash
gcc src/c/main.c -o hermes

```

* **Via Makefile :**
```bash
make all

```

**2. Exécution :**

* **Sur Windows (PowerShell) :**
```powershell
.\hermes.exe
```

* **Sur Linux / Mac :**
```bash
./hermes
```

*(Optionnel)* Pour nettoyer l'espace de travail et supprimer l'exécutable généré : `make clean` ou `rm hermes`.

---

## 🧠 Détails Techniques

### Architecture du Modèle (MLP)

* **Input Layer** : 60 neurones (Fenêtre glissante des 60 derniers jours).
* **Hidden Layer** : 32 neurones (Activation ReLU).
* **Output Layer** : 1 neurone (Prix prédit, Activation Linéaire).

### Gestion des Données

* **Normalisation** : MinMax Scaling (0-1).
* **Persistance** : Les valeurs `Min` et `Max` utilisées lors de l'entraînement sont sauvegardées dans l'en-tête du fichier `.bin` pour garantir que le moteur C normalise les données exactement comme le script Python.

---

## 📜 Licence
Projet open-source sous licence **MIT**.

## Version
**v1.1.0**