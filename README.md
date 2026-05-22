# 🎫 Ticket Routing — Priorisation et Routage Intelligent de Tickets Support

> Projet de fin d'études — Big Data & Intelligence Artificielle  
> Pipeline ML complet : prétraitement · modèles maison & préentraîné · API FastAPI · Interface Streamlit

***

##  Présentation du projet

Ce projet propose un système de **classification automatique de tickets support** basé sur le deep learning. À partir du texte d'un ticket et de métadonnées client, le système prédit simultanément :

- **La priorité** du ticket : `Critical`, `High`, `Medium`, `Low`
- **La catégorie de routage** : `support_technique`, `facturation`, `information_produit`, `retour_produit`, `annulation`, `autre`

L'objectif métier est d'automatiser le tri des tickets entrants pour réduire le temps de traitement et orienter chaque demande vers le bon service.

***

##  Structure du projet

```
ticket-routing/
├── data/
│   ├── raw/                  ← données brutes (CSV original)
│   └── processed/            ← données nettoyées et enrichies
│
├── notebooks/
│   ├── 01_exploration.ipynb          ← analyse exploratoire (EDA)
│   ├── 02_preprocessing.ipynb        ← nettoyage et feature engineering
│   ├── 03_modele_maison.ipynb        ← modèle custom BiLSTM
│   ├── 04_modele_pretrained.ipynb    ← modèle TF-IDF + Dense
│   └── 05_comparative_study.ipynb   ← comparaison des deux approches
│
├── src/
│   ├── api/
│   │   └── app.py            ← API FastAPI (backend)
│   ├── data/
│   │   ├── clean_data.py     ← nettoyage des données
│   │   └── build_features.py ← feature engineering + splits
│   ├── models/
│   │   ├── common.py         ← utilitaires partagés (df_to_inputs)
│   │   ├── model_custom.py   ← architecture BiLSTM (modèle maison)
│   │   ├── model_pretrained.py ← architecture TF-IDF + Dense
│   │   ├── train_priority.py ← entraînement tâche priority
│   │   ├── train_routing.py  ← entraînement tâche routing
│   │   ├── evaluate.py       ← évaluation des modèles
│   │   └── predict.py        ← pipeline de prédiction
│   └── utils/
│       └── config.py         ← chemins et constantes globales
│
├── models/                   ← modèles .keras sauvegardés
│   ├── priority_custom.keras
│   ├── priority_pretrained.keras
│   ├── routing_custom.keras
│   ├── routing_pretrained.keras
│   ├── priority_class_names.pkl
│   └── routing_class_names.pkl
│
├── frontend/
│   └── streamlit_app.py      ← interface utilisateur Streamlit
│
├── reports/
│   └── figures/              ← graphiques générés (courbes, matrices)
│
├── requirements.txt
└── README.md
```

***

##  Approche technique

### Données

Le dataset utilisé est un fichier CSV de tickets support client contenant :
- le texte du ticket (`Ticket Description`)
- des métadonnées : âge et genre du client, produit acheté, type et canal du ticket
- deux labels cibles : `Ticket Priority` et une catégorie de routage construite

### Preprocessing

- Nettoyage du texte (minuscules, suppression des caractères spéciaux)
- Encodage des variables catégorielles (`StringLookup`)
- Normalisation des variables numériques (`Normalization`)
- Création des labels de routage par règles métier
- Séparation train/test stratifiée (80/20)

### Modèle 1 — Custom (BiLSTM)

Architecture multimodale Keras Functional API :
- **Branche texte** : `TextVectorization` (int) → `Embedding(128)` → `Bidirectional LSTM(64)`
- **Branches catégorielles** : `StringLookup` one-hot → `Flatten`
- **Branche numérique** : `Normalization`
- **Fusion** : `Concatenate` → `Dense(256)` → `BatchNorm` → `Dropout(0.4)` → `Dense(128)` → `Softmax`

### Modèle 2 — Préentraîné style (TF-IDF + Dense)

Même architecture multimodale mais avec une branche texte différente :
- **Branche texte** : `TextVectorization` (TF-IDF) → `Flatten` → `Dense(256)` → `BatchNorm` → `Dropout`
- **Fusion identique** au modèle custom

### Entraînement

- Optimiseur : `Adam(lr=1e-3)`
- Loss : `sparse_categorical_crossentropy`
- Callbacks : `EarlyStopping(patience=4)` + `ReduceLROnPlateau(factor=0.5)`
- `class_weight='balanced'` pour gérer le déséquilibre des classes

### Résultats (Comparative Study)

| Tâche | Modèle | Accuracy | F1 Macro |
|---|---|---|---|
| Priority | Custom BiLSTM | ~0.26 | ~0.25 |
| Priority | TF-IDF Dense | ~0.28 | ~0.26 |
| Routing | Custom BiLSTM | — | — |
| Routing | TF-IDF Dense | — | — |

> Les performances sur la tâche Priority sont limitées par la faible corrélation entre le texte et les labels de priorité dans ce dataset. La tâche Routing donne de meilleurs résultats car le vocabulaire est directement lié à la catégorie métier.

***

##  Installation et lancement

### Prérequis

- Python 3.10+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer le backend FastAPI

```bash
uvicorn src.api.app:app --reload --port 8000
```

Documentation interactive disponible sur : [http://localhost:8000/docs](http://localhost:8000/docs)

### Lancer le frontend Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

Interface disponible sur : [http://localhost:8501](http://localhost:8501)

***

##  API — Endpoints

### `GET /`
Vérifie que l'API est en ligne.

**Réponse :**
```json
{"message": "API de priorisation et routage intelligent de tickets"}
```

### `GET /health`
Vérifie l'état du service.

**Réponse :**
```json
{"status": "ok"}
```

### `POST /predict`
Prédit la priorité et la catégorie de routage d'un ticket.

**Body (JSON) :**
```json
{
  "text": "My laptop screen is broken and I need urgent help",
  "Customer_Gender": "Male",
  "Product_Purchased": "Laptop",
  "Ticket_Type": "Technical support",
  "Ticket_Channel": "Email",
  "Customer_Age": 35,
  "model_name": "custom"
}
```

**Réponse :**
```json
{
  "predicted_priority": "Critical",
  "predicted_routing": "support_technique",
  "model_used": "custom"
}
```

> Le champ `model_name` accepte `"custom"` (BiLSTM) ou `"pretrained"` (TF-IDF).

***

##  Interface Streamlit

L'interface permet de :
- Saisir un ticket complet via un formulaire
- Choisir le modèle à utiliser (`custom` ou `pretrained`)
- Visualiser la priorité avec un code couleur (🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Low)
- Visualiser la catégorie de routage prédite
- Consulter les détails de la requête envoyée à l'API

***

##  Dépendances principales

```
tensorflow>=2.15
fastapi
uvicorn
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
pydantic
requests
```

***

