# README.md

## Introduction

Ce projet a pour but de créer un pipeline de **classification multi-étiquettes supervisée** sur des données textuelles, en utilisant des techniques avancées de représentation comme **BoW**, **TF-IDF**, **Word2Vec**, **Doc2Vec**, **BERT**, et **Universal Sentence Encoder (USE)**. 

Les **données** sont **téléchargées et traitées en local**, et les modèles sont entraînés et évalués sur plusieurs configurations de tags (top 15 et top 50). Le **meilleur modèle** pour chaque configuration (basé sur TF-IDF et AdaBoost) est sauvegardé pour une utilisation dans l'API.

L'**API**, quant à elle, permet de prédire les tags pour une phrase donnée, selon le nombre de tags souhaités et la configuration de tags choisie (top 15 ou top 50). 

Le pipeline ne contient pas de surveillance de **model drift** ou de **concept drift** dans ce projet.

## Structure des Dossiers et Fichiers

### Dossiers

- **vectorizers/** : Ce dossier contient les fichiers liés aux **vectoriseurs** utilisés dans pour l'api projet, tels que **TF-IDF** pour la versoin actuelle . Ces fichiers sont générés et sauvegardés en local après le prétraitement des données textuelles.

### Fichiers Importants

- **app.py** : Ce fichier contient le **code de l'API**. Il permet de faire des prédictions en fonction d'une phrase d'entrée, du choix de la configuration de tags (top 15 ou top 50) et du nombre de tags à prédire. L'API renvoie les tags correspondants à la phrase.

- **mlb_top_15.pkl** & **mlb_top_50.pkl** : Ces fichiers contiennent les encodages multi-étiquettes (`MultiLabelBinarizer`) pour les configurations de tags top 15 et top 50. Ils sont utilisés pour convertir les étiquettes en vecteurs multi-étiquettes pour l'entraînement et la prédiction.

- **model_top_15.pkl** & **model_top_50.pkl** : Ce sont les **meilleurs modèles** basés sur **TF-IDF avec AdaBoost**, entraînés respectivement sur les configurations top 15 et top 50. Ces modèles sont utilisés pour prédire les tags via l'API.

- **pca_top_15.pkl** & **pca_top_50.pkl** : Modèles **PCA** utilisés pour la réduction de dimensionnalité des données pour les configurations top 15 et top 50.

- **requirements.txt** : Ce fichier contient la liste des **dépendances** nécessaires à l'exécution du projet. Utilisez la commande ci-dessous pour installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

- **Procfile** : Fichier de configuration pour déployer l'API sur **Heroku** ou tout autre service cloud compatible. Il définit comment démarrer l'application en production.

- **Note Technique.docx** : Ce document contient une étude détaillée sur l'architecture du pipeline MLOps, le prétraitement des données, l'entraînement et l'évaluation des modèles.

## Instructions pour Exécuter le Projet

1. **Cloner le Dépôt** : Clonez ce dépôt Git sur votre machine locale :
   ```bash
   git clone <repository-url>
   ```

2. **Installer les Dépendances** : Installez les bibliothèques nécessaires à partir du fichier `requirements.txt` :
   ```bash
   pip install -r requirements.txt
   ```

3. **Entraînement et Prétraitement en Local** : Le script `app.py` ne contient que le code pour l'API. Le **traitement des données**, l'**entraînement des modèles** et l'**évaluation des performances** sont déjà faits en local, et les **meilleurs modèles** pour les configurations top 15 et top 50 ont été sauvegardés dans des fichiers `.pkl`.

4. **Exécuter l'API en Local** :
   Lancez l'application API pour commencer à faire des prédictions :
   ```bash
   python app.py
   ```

5. **Requêtes API** : L'API accepte des requêtes où vous fournissez une phrase, le choix de la configuration de tags (top 15 ou top 50), ainsi que le nombre de tags à prédire. L'API renverra les tags correspondants à la phrase.

6. **Déploiement sur Heroku** : Utilisez le fichier `Procfile` pour déployer l'API sur **Heroku** ou un autre service cloud compatible. Seuls les meilleurs modèles TF-IDF avec AdaBoost (pour top 15 et top 50) seront utilisés pour les prédictions.

## Fonctionnement de l'API

L'API reçoit en entrée :
- Une **phrase** : le texte à analyser.
- Le **top X tags** : choix entre la configuration top 15 ou top 50.
- Le **nombre de tags à prédire** : le nombre de tags que l'API doit renvoyer pour cette phrase.

L'API retourne les tags les plus pertinents en fonction du modèle **TF-IDF avec AdaBoost**.

## Fonctionnalités du Projet

- **Représentation Textuelle** : Le projet utilise des méthodes avancées de représentation comme **BoW**, **TF-IDF**, **Word2Vec**, **Doc2Vec**, **BERT**, et **USE**. Le modèle sélectionné pour l'API est basé sur **TF-IDF**.

- **Classification Multi-Étiquettes** : Le modèle d'apprentissage utilisé est **AdaBoost**, qui a été entraîné pour prédire plusieurs étiquettes à partir d'un texte.

- **API de Prédiction** : Le code `app.py` fournit une API REST qui permet d'entrer une phrase et d'obtenir les tags les plus probables, en fonction du nombre de tags souhaité et de la configuration choisie (top 15 ou top 50).

## Contributions

Les contributions sont les bienvenues. Merci de vous assurer que votre code respecte les standards **PEP 8** et d'inclure des tests unitaires lorsque c'est nécessaire. Vous pouvez soumettre des améliorations via des issues ou des pull requests.

---

Ce fichier `README.md` fournit une vue d'ensemble du projet, les étapes pour exécuter l'API et des instructions pour le déploiement en local ou sur Heroku.
