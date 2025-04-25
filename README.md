# Information Spotting - Recherche de Contenu Visuel dans des Documents

Ce projet propose un système d’indexation et de recherche automatique d’informations visuelles basé sur la correspondance de descripteurs SIFT extraits d’images et de vidéos. Il permet d’identifier un document cible et de localiser précisément une région correspondante dans celui-ci à partir d’une vidéo.

## Fonctionnalités

- Extraction des points d’intérêt SIFT pour chaque image (document).
- Indexation par *table de hachage* pour une recherche rapide et robuste.
- Extraction de requêtes depuis une *vidéo utilisateur*.
- Appariement entre la requête et les documents indexés.
- Localisation de la *région pertinente* dans le document correspondant.
- Interface utilisateur via *Streamlit*.

## Démonstration des Résultats

L’interface affiche les trois volets suivants :

| Query (extrait vidéo) | Document Retrieval | Region Spotted |
|------------------------|--------------------|----------------|
| ![query](data/videos/frame_query.png) | ![doc](data/documents/0002.jpg) | ![region](outputs/region_result.png) |

> Le système est capable d’identifier visuellement un extrait de texte issu d’un document filmé et de retrouver ce passage dans un ensemble d’images de documents.

## Arborescence du Projet
