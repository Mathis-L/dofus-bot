# Dofus Bot Bûcheron & Alchimiste avec Vision par Ordinateur (YOLO)

⚠️ **AVERTISSEMENT IMPORTANT** ⚠️
L'utilisation de bots, y compris celui-ci, est **strictement interdite** par les Conditions Générales d'Utilisation (CGU) de Dofus. L'utilisation de ce projet expose votre compte à un **risque élevé de bannissement permanent**. Ce projet est partagé à des fins éducatives et techniques uniquement. L'auteur décline toute responsabilité quant aux conséquences de son utilisation. **Utilisez-le à vos risques et périls.**

## Table des Matières
- [Dofus Bot Bûcheron \& Alchimiste avec Vision par Ordinateur (YOLO)](#dofus-bot-bûcheron--alchimiste-avec-vision-par-ordinateur-yolo)
  - [Table des Matières](#table-des-matières)
  - [Introduction](#introduction)
  - [Fonctionnalités Clés](#fonctionnalités-clés)
  - [Technologies Utilisées](#technologies-utilisées)
  - [La Recette : Comment ce Bot a été Conçu](#la-recette--comment-ce-bot-a-été-conçu)
    - [Étape 1: Collecte de Données (Captures d'Écran)](#étape-1-collecte-de-données-captures-décran)
    - [Étape 2: Annotation des Images avec Label Studio (ou Roboflow)](#étape-2-annotation-des-images-avec-label-studio-ou-roboflow)
    - [Étape 3: Entraînement du Modèle YOLO (Transfer Learning avec Roboflow et Colab)](#étape-3-entraînement-du-modèle-yolo-transfer-learning-avec-roboflow-et-colab)
    - [Étape 4: Développement de l'Algorithme de Botting](#étape-4-développement-de-lalgorithme-de-botting)
  - [Installation](#installation)
  - [Configuration et Utilisation](#configuration-et-utilisation)
  - [Aperçu de la Structure du Code](#aperçu-de-la-structure-du-code)
  - [Pistes d'Amélioration](#pistes-damélioration)
  - [Portabilité et Limitations Multi-Plateformes](#portabilité-et-limitations-multi-plateformes)

## Introduction
Ce projet détaille la création d'un bot pour le jeu Dofus, capable d'exercer les métiers de bûcheron et d'alchimiste de manière automatisée. L'approche principale repose sur la vision par ordinateur (Computer Vision) pour identifier les ressources récoltables à l'écran, en utilisant un modèle YOLO entraîné spécifiquement.

Ce README est structuré comme une "recette", expliquant pas à pas les étapes qui ont mené à la réalisation de ce bot.

## Fonctionnalités Clés
*   Récolte automatisée des ressources de bûcheron et d'alchimiste.
*   Détection des ressources à l'écran via un modèle YOLO personnalisé.
*   Navigation intelligente entre les cartes du jeu.
*   Gestion de la zone de farm et des déplacements.
*   Simulation de mouvements et clics de souris "humains".
*   Détection d'agressions par des monstres et mise en pause du bot.
*   Gestion basique de l'inventaire (pods).
*   Interface graphique (Tkinter) pour la configuration et le lancement.
*   Sauvegarde de l'état des cartes (accessibilité, ressources) pour optimiser les parcours.

## Technologies Utilisées
*   **Python 3**: Langage de programmation principal.
*   **YOLO (Ultralytics)**: Modèle de détection d'objets en temps réel pour identifier les ressources. (Utilisé via la librairie `ultralytics`, compatible avec les formats d'export de Roboflow comme "YOLOv11" qui correspond à une structure spécifique pour Ultralytics).
*   **OpenCV (`cv2`)**: Bibliothèque pour le traitement d'images et la manipulation des captures d'écran.
*   **MSS**: Capture d'écran rapide et efficace.
*   **PyAutoGUI**: Contrôle de la souris et du clavier pour les interactions en jeu.
*   **HumanCursor**: Simulation de mouvements de souris plus naturels.
*   **Keyboard**: Gestion des raccourcis clavier (ex: arrêt d'urgence).
*   **Tkinter**: Création de l'interface graphique utilisateur.
*   **Label Studio / Roboflow.com**: Outils d'annotation d'images et de gestion de datasets. Pour ce projet, un dataset est disponible sur Roboflow.
*   **Google Colab (ou environnement similaire)**: Pour l'entraînement du modèle YOLO (transfer learning), facilité par le notebook `Train_YOLO_Models_Dofus.ipynb`.

## La Recette : Comment ce Bot a été Conçu

Voici les étapes suivies pour développer ce bot Dofus :

### Étape 1: Collecte de Données (Captures d'Écran)
La base de tout système de vision par ordinateur est un bon jeu de données.
1.  **Prise de captures d'écran en jeu** :
    *   Parcourir différentes cartes de Dofus contenant les ressources ciblées (frênes, trèfles, orties, etc.).
    *   Capturer des images dans diverses conditions : différents moments de la journée en jeu (si l'éclairage change), différentes configurations de ressources, avec et sans l'option "Afficher les ressources récoltables" (touche 'Y' par défaut).
    *   L'objectif est d'obtenir un ensemble d'images variées pour que le modèle apprenne à généraliser.
    *   Plus le modèle a d'exemples plus il sera efficace.

### Étape 2: Annotation des Images avec Label Studio (ou Roboflow)
Une fois les images collectées, elles doivent être "labellisées" pour que le modèle sache quoi chercher. Si vous utilisez le dataset fourni, cette étape a déjà été réalisée.
1.  **Option A: Utiliser Label Studio**
    *   Importation des images dans Label Studio ([https://labelstud.io/](https://labelstud.io/)).
    *   Création des labels : Définir les classes de ressources à détecter (ex: `Frene`, `Chene`, `Trefle`, `Ortie`, etc.).
    *   Annotation (Bounding Boxes) : Pour chaque image, dessiner des boîtes englobantes (bounding boxes) autour de chaque instance de ressource. Associer chaque boîte au label correspondant.
        *   Soyez précis et cohérent dans l'annotation.
    *   Exportation des annotations : Exporter les annotations au format compatible YOLO.
2.  **Option B: Utiliser Roboflow (Recommandé avec le dataset fourni)**
    *   Roboflow ([https://roboflow.com/](https://roboflow.com/)) permet de gérer, annoter, augmenter et exporter des datasets.
    *   Un dataset pré-annoté pour ce projet est disponible ici : [Dofus Resource Detection Dataset on Roboflow](https://universe.roboflow.com/mathisl-qvljq/dofus-resource-detection)

### Étape 3: Entraînement du Modèle YOLO (Transfer Learning avec Roboflow et Colab)
Avec les données prêtes (collectées et annotées par vous-même ou via le dataset Roboflow), on peut entraîner le modèle de détection. Le notebook `Train_YOLO_Models_Dofus.ipynb` est conçu pour cela.

1.  **Obtention du Dataset depuis Roboflow (Recommandé)**:
    *   Accédez au dataset sur Roboflow :  [Dofus Resource Detection Dataset on Roboflow](https://universe.roboflow.com/mathisl-qvljq/dofus-resource-detection).
    *   Dans votre projet Roboflow (ou celui fourni), après avoir sélectionné une version du dataset :
        1.  Cliquez sur "Export Dataset".
        2.  Choisissez le format **"YOLOv11"** (ou le format le plus récent compatible Ultralytics/YOLOv5+, par exemple "YOLO v8"). Ce format inclut typiquement :
            *   Des dossiers `train/`, `valid/` (et parfois `test/`) contenant chacun des sous-dossiers `images/` et `labels/`.
            *   Un fichier `data.yaml` déjà configuré pour votre dataset, pointant vers ces dossiers.
        3.  Choisissez l'option "download zip to computer". Vous obtiendrez un fichier ZIP (par exemple, `Dofus-Resources-1.zip`).

2.  **Entraînement avec Google Colab et le Notebook `Train_YOLO_Models_Dofus.ipynb`**:
    *   Ouvrez Google Colab ([https://colab.research.google.com/](https://colab.research.google.com/)).
    *   Téléchargez (upload) le notebook `Train_YOLO_Models_Dofus.ipynb` dans votre environnement Colab.
    *   **Suivez les instructions détaillées dans le notebook `Train_YOLO_Models_Dofus.ipynb`**.

### Étape 4: Développement de l'Algorithme de Botting
C'est ici que le script Python fourni (`BotDofus.py`) entre en jeu. Il orchestre la détection, l'interaction et la navigation.

1.  **Initialisation et Configuration (`BotInterface`, `HarvestBot.__init__`)**:
    *   Une interface Tkinter (`BotInterface`) permet à l'utilisateur de configurer :
        *   Les types de ressources à récolter.
        *   La position de départ du personnage (`(x, y)` en coordonnées de carte Dofus).
        *   La carte actuelle (ex: "Incarnam", "Astrub").
        *   Une zone de restriction pour le farm.
        *   Les pods disponibles.
    *   Le `HarvestBot` est initialisé avec ces paramètres et charge le modèle YOLO (`my_model.pt`).

2.  **Boucle Principale de Récolte (`HarvestBot.run_harvest`)**:
    *   **Focus sur le jeu** (`focus_on_game`): Clique sur la mini-carte pour s'assurer que la fenêtre Dofus est active.
    *   **Affichage des ressources** : Simule l'appui sur la touche 'Y' pour rendre les ressources visibles.
    *   **Capture d'écran** (`capture_screen` avec `mss`): Prend une capture de l'écran du jeu.
    *   **Détection de Ressources (`process_detections`)**:
        *   L'image capturée est passée au modèle YOLO (`self.model(img_path)`).
        *   Le modèle retourne les boîtes englobantes et les classes des ressources détectées.
        *   Les coordonnées du centre de chaque ressource pertinente (celles sélectionnées par l'utilisateur et dans la zone de farm) sont extraites.
        *   Les informations sur les ressources présentes sur la carte sont mises à jour dans `GameMap`.
    *   **Interaction avec les Ressources**:
        *   Si des ressources sont détectées, le bot simule des clics "humains" (`human_like_click`) sur chaque ressource.
        *   Un thread de surveillance (`monitor_monster_attack`) est lancé pour détecter les agressions de monstres (en vérifiant si l'écran devient noir, signe d'un combat).
        *   Si un monstre est détecté (`monster_detected`), le bot se met en pause, alerte l'utilisateur et attend une intervention manuelle (appui sur 'Entrée' après le combat).
        *   Les pods sont mis à jour après chaque récolte (`add_to_inventory`).
    *   **Vérification de l'inventaire** (`is_inventory_full`): Si l'inventaire est plein, une alerte est émise.
    *   **Changement de Carte (`change_map`)**:
        *   La classe `GameMap` gère la logique de déplacement.
        *   Elle stocke les informations sur les cartes visitées, leur accessibilité depuis d'autres cartes, et le nombre de visites. Ces données sont sauvegardées dans un fichier (ex: `Incarnam_map.txt`).
        *   `get_possible_moves` détermine les directions de changement de carte possibles depuis la position actuelle, en privilégiant les cartes moins visitées et celles situées dans la zone de restriction définie.
        *   Le bot clique sur le bord de l'écran correspondant à la direction choisie.
        *   `detect_map_change` vérifie si le changement de carte a réussi (en détectant l'écran noir de transition).
        *   L'accessibilité entre les cartes est mise à jour dans `GameMap`.
        *   La position actuelle du bot (`self.current_position`) est mise à jour.
    *   La boucle recommence.

3.  **Arrêt du Bot (`HarvestBot.stop`, `BotInterface.stop_bot`)**:
    *   L'utilisateur peut arrêter le bot via une touche de raccourci (`*` par défaut).
    *   L'état actuel de la `GameMap` (informations sur les cartes) est sauvegardé.

## Installation
1.  Clonez ce dépôt (ou copiez les fichiers du projet).
2.  Ouvrez un terminal ou une invite de commande dans le dossier du projet.
3.  (Recommandé) Créez un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    venv\Scripts\activate    # Sur Windows
    ```
4.  Installez les dépendances. Le fichier `requirements.txt` contient les librairies suivantes :
    ```
    torch
    mss
    numpy
    opencv-python
    keyboard
    pyautogui
    humancursor
    ultralytics
    ```
    Puis installez-les :
    ```bash
    pip install -r requirements.txt
    ```
    *Note: L'installation de `torch` peut nécessiter une commande spécifique en fonction de votre configuration CUDA si vous souhaitez utiliser le GPU. Consultez le site de PyTorch. L'entraînement YOLO sur GPU avec `ultralytics` nécessite également une installation PyTorch compatible CUDA.*
5.  Assurez-vous que le fichier `my_model.pt` (votre modèle YOLO entraîné, obtenu à l'Étape 3) est dans le même répertoire que le script principal (ex: `BotDofus.py`).
6.  Les fichiers de carte (ex: `Incarnam_map.txt`) seront créés automatiquement lors de la première utilisation si absents.
7.  (Optionnel) Si vous souhaitez entraîner votre propre modèle, assurez-vous d'avoir le notebook `Train_YOLO_Models_Dofus.ipynb` et suivez l'Étape 3.

## Configuration et Utilisation
1.  Lancez le jeu Dofus et connectez-vous avec votre personnage. Positionnez-le sur une carte de départ.
2.  Exécutez le script principal (par exemple, `BotDofus.py`) :
    ```bash
    python BotDofus.py
    ```
3.  L'interface graphique de configuration s'ouvrira :
    *   **Choisissez les classes** : Cochez les ressources que le bot doit récolter (assurez-vous que votre `my_model.pt` a été entraîné pour ces classes).
    *   **Position de départ** : Entrez les coordonnées X et Y de la carte où se trouve votre personnage.
    *   **Choisissez la carte** : Sélectionnez le nom de la zone principale (ex: "Incarnam", "Astrub"). Cela permet de sauvegarder la carte pour être plus efficace dans le futur.
    *   **Zone de restriction** : Définissez les coordonnées (X1, Y1) du coin supérieur gauche et (X2, Y2) du coin inférieur droit de la zone où le bot doit rester.
    *   **Pods** : Entrez vos pods actuellement utilisés et vos pods maximum.
4.  Cliquez sur "Lancer le bot".
5.  Le bot commencera à travailler. L'interface affichera son état ("Bot running") et les coordonnées actuelles.
6.  Pour arrêter le bot, appuyez sur la touche `*` (astérisque) de votre clavier. Un message "Bot arrêté" s'affichera.

## Aperçu de la Structure du Code
Le script Python fourni est organisé en trois classes principales :

*   **`GameMap`**:
    *   Gère les données de la carte (coordonnées, ressources présentes, accessibilité entre les cartes).
    *   Charge et sauvegarde ces données dans un fichier texte (format JSON).
    *   Calcule les mouvements possibles et aide à la décision de la prochaine carte à visiter.

*   **`HarvestBot`**:
    *   Contient la logique principale du bot.
    *   Initialise le modèle YOLO (`my_model.pt`) et les paramètres de récolte.
    *   Gère la boucle de capture d'écran, détection de ressources, interaction (clics), et changement de carte.
    *   Implémente la détection d'agression, la gestion des pods, et les alertes sonores.
    *   Utilise `HumanCursor` et `PyAutoGUI` pour des interactions plus naturelles.

*   **`BotInterface`**:
    *   Crée et gère l'interface graphique utilisateur avec Tkinter.
    *   Permet à l'utilisateur de configurer et de lancer/arrêter le bot.
    *   Affiche des informations sur l'état du bot.

## Pistes d'Amélioration
*   **Gestion des combats plus avancée** : Plutôt que de simplement pauser, implémenter une logique de combat basique ou une fuite. (Beaucoup plus complexe ...)
*   **Gestion de la banque** : Aller déposer les ressources à la banque lorsque l'inventaire est plein.
*   **Anti-détection plus poussée** : Varier davantage les temps d'attente, les trajectoires de souris, etc. (augmente la complexité et le risque).
*   **Optimisation des trajets** : Algorithmes plus sophistiqués pour choisir la ressource ou la prochaine carte.
*   **Gestion des erreurs améliorée** : Gérer plus de cas imprévus (déconnexions, fenêtres pop-up du jeu).
*   **Support de plusieurs métiers/ressources** : Améliorer le modèle avec plus de données du jeu et adapter la logique de récolte.
*   **Détection dynamique de la fenêtre de jeu**: Plutôt que de se baser sur des captures d'écran complètes, identifier et capturer uniquement la fenêtre Dofus.

## Portabilité et Limitations Multi-Plateformes

Ce bot a été initialement développé et testé principalement sous **Windows**. Bien que beaucoup des bibliothèques utilisées (Python, PyTorch, OpenCV, Tkinter) soient multi-plateformes, plusieurs aspects du code actuel le rendent **non fonctionnel "out-of-the-box" sur d'autres systèmes d'exploitation (macOS, Linux) ou même sur différentes configurations Windows sans modifications.**

**Principaux Points de Non-Portabilité à Considérer :**
1.  **Alertes Sonores (`winsound`)**: La bibliothèque `winsound` est spécifique à Windows. Pour une fonctionnalité sonore sur d'autres OS, elle devrait être remplacée par une alternative multi-plateforme (ex: `playsound`, `simpleaudio`) ou rendue conditionnelle.
2.  **Coordonnées d'Écran Fixes**:
    *   Les positions pour cliquer afin de changer de map (bords de l'écran), la zone de la mini-carte pour le focus, et potentiellement d'autres clics fixes sont actuellement "hardcodées" (définies par des valeurs numériques fixes dans le code, souvent basées sur une résolution d'écran spécifique comme 1920x1080).
    *   Ces coordonnées dépendent crucialement de la **résolution de votre écran**, de la **taille et position de la fenêtre Dofus**. Elles ne fonctionneront très probablement pas correctement sur une autre machine ou si votre configuration d'affichage change.
    *   **Modification Nécessaire**: Ces coordonnées devraient être rendues configurables par l'utilisateur (par exemple, via l'interface graphique ou un fichier de configuration) ou, idéalement, détectées dynamiquement par rapport à la fenêtre du jeu.
3.  **Sélection du Moniteur de Capture (`mss`)**: Le script utilise `mss` pour la capture. La configuration du moniteur (ex: `monitors[1]`) peut nécessiter un ajustement si Dofus est sur un autre écran ou si la numérotation des moniteurs diffère.
    *   **Modification Nécessaire**: Permettre à l'utilisateur de spécifier quel moniteur capturer, ou tenter de détecter la fenêtre Dofus et capturer cette région spécifique.
4.  **Contrôle Souris/Clavier (`pyautogui`, `keyboard`)**: Les bibliothèques comme `pyautogui` et `keyboard` peuvent se comporter différemment ou nécessiter des permissions spécifiques selon l'OS (ex: permissions d'accessibilité sur macOS, droits root/sudo pour les hotkeys globales sur certaines configurations Linux). La touche d'arrêt `*` peut aussi être interprétée différemment ou ne pas fonctionner globalement.

**En résumé :** Pour utiliser ce bot sur une machine autre que celle pour laquelle il a été finement ajusté (surtout concernant les coordonnées d'écran et la configuration des moniteurs) ou sur un OS autre que Windows, des **modifications significatives du code sont probablement nécessaires.** L'effort requis peut varier de quelques ajustements de configuration à des refontes de certaines parties du code relatives aux interactions avec l'écran et le système.
