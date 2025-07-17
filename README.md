# Étude Longitudinale de l'Évolution d'une Tumeur

**Auteur:** Paul Haardt

# Il faut installer les requirements dans requirements.txt !

## 0. Déroulement

Plusieurs fenêtres interactives présentent le contenu du projet.
Il suffit de les fermer pour passer à l'étape suivante.

La dernière étape consiste en une visualisation 3D de la tumeur.
Déplacer la souris verticalement et horizontalement affectent respectivement la visibilité de chaque tumeur.
La légende explique quelle tumeur est visible.
L'interaction est de type "trackball", il n'y a donc qu'à maintenir le clic pour tourner autour dans la visualisation.
La molette permet de régler le zoom.

## 1. Choix Techniques

### Recalage d'Images
- **Algorithme:** Recalage 2D slice-by-slice avec transformation de translation
- **Métrique:** Moyenne des carrés (Mean Squares)
- **Optimiseur:** Gradient Descent régulier avec 100 itérations
- **Justification:** Comme vu en cours

### Segmentation
- **Méthode:** Croissance de région 3D (Connected Threshold)
- **Paramètres:** Seed point (90, 70, 50), seuils ±100/±50 autour de l'intensité seed
- **Pré-traitement:** Lissage anisotrope (5 itérations, conductance=3)
- **Zone d'analyse:** Coupes 41-59 (région tumorale principale)
- **Justification** La détection de la tumeur n'était pas assez stable sans aide initiale.

### Analyse des Changements
- **Métriques volumétriques:** Volume, surface, centroïde
- **Métriques de similarité:** Coefficient de Dice
- **Analyse spatiale:** Changements slice-by-slice

## 2. Difficultés Rencontrées

### Problèmes de Compatibilité
- **NumPy 2.x:** Incompatibilité avec scipy/scikit-image
- **Solution:** Implémentation pure NumPy des fonctions morphologiques

### Optimisation de l'Analyse
- **Problème:** Région tumorale concentrée sur certaines coupes
- **Solution:** Focus sur les coupes 41-59

## 3. Résultats Obtenus

### Évolution Volumétrique
- **Volume tumoral:** 4725 → 5210 mm³ (+485 mm³)
- **Croissance:** +10.3% 
- **Interprétation:** Augmentation significative du volume tumoral

### Évolution Morphologique
- **Surface:** 3596 → 3082 mm² (-514 mm²)
- **Changement:** -14.3%
- **Interprétation:** La tumeur devient plus compacte (volume plus gros, surface plus petite)

### Analyse Spatiale
- **Déplacement du centre:** 2.8 voxels
- **Coefficient de Dice:** 53.8%
- **Interprétation:** Déplacement modéré, chevauchement moyen entre les deux temps

### Distribution des Changements
- **Coupes en croissance:** 15/19 (79%)
- **Coupes en rétrécissement:** 4/19 (21%)
- **Croissance maximale:** 91 voxels (coupe 48)
- **Rétrécissement maximal:** 39 voxels

### Évolution d'Intensité
- **Intensité moyenne:** 691.7 → 684.7 (-7.0)
- **Interprétation:** Légère diminution de l'intensité du signal

## 4. (bonus) interprétation

**La tumeur a grossi :**
- Le volume a augmenté de 10%
- La tumeur a légèrement bougé

**La croissance n'est pas uniforme :**
- 15 coupes montrent une expansion
- 4 coupes montrent un rétrécissement
- La croissance est plus marquée au centre (coupes 47-48)

## 5. Limitations et Perspectives

### Limitations Actuelles
- Segmentation semi-automatique (dépendante du seed point)
- Analyse limitée aux coupes principales
- Segmentation automatique par machine learning
