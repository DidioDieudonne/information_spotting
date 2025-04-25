Documents# Information Spotting - Recherche de Contenu Visuel dans des Documents

Ce projet met en œuvre un système de *spotting d'information* à partir d'une *requête vidéo. Il permet d'identifier automatiquement le **document* le plus similaire et de *localiser visuellement* la *région d'intérêt* présente dans le document, à l'aide de descripteurs SIFT et d'une table de hachage.

---

## Fonctionnalités

- Extraction des *points d’intérêt* et *descripteurs SIFT* pour chaque document image.
- *Indexation* des documents via une *table de hachage* optimisée.
- Traitement automatique d’une vidéo utilisateur pour extraire des requêtes visuelles.
- *Matching robuste* entre la requête vidéo et les documents.
- *Localisation précise* de la région de la requête dans le document retrouvé.
- Interface utilisateur *intuitive via Streamlit*.

---

## Résultat Visuel

L'application retourne trois résultats affichés côte à côte :  

| Query (extrait vidéo) | Document Retrieval | Region Spotted |
|------------------------|--------------------|----------------|
| ![Query](data/videos/frame_query.png) | ![Document](data/documents/0002.jpg) | ![Region](data/outputs/region_result.png) |

> Le système repère efficacement un extrait textuel filmé dans une vidéo et localise sa position exacte dans un document image correspondant.

---

## Arborescence du Projet

information_spotting/
├── app_information_spotting.py       # Application principale Streamlit
├── extract_features.py               # Extraction descripteurs SIFT
├── indexer.py                        # Indexation par table de hachage
├── matcher.py                        # Matching et localisation
├── query_video.py                    # Requêtes vidéo
├── requirements.txt                  # Dépendances Python
└── data/
├── documents/                    # Images à indexer
├── videos/                       # Vidéos de requêtes
├── descriptors/                  # Descripteurs SIFT sauvegardés
└── outputs/                      # Résultats visuels (region spotted)

---

## Installation

Assurez-vous d’avoir *Python 3.10+* et *pip* installés.

```bash
git clone https://github.com/ton_utilisateur/information_spotting.git
cd information_spotting
pip install -r requirements.txt



⸻

Lancer l’application

streamlit run app_information_spotting.py

Cela ouvrira l’interface web dans votre navigateur.

⸻

Aperçu de l’interface


⸻

### Pour l’image result_demo.jpg :
Tu peux prendre une *capture d’écran* de l’interface Streamlit avec les trois colonnes Query, Document, Region Spotted et la sauvegarder dans le dossier docs/ sous le nom result_demo.jpg.
