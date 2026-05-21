# Customer Support Ticket Prioritization and Intelligent Routing

Projet TensorFlow/Keras de classification **texte + tabulaire** pour un cas d’usage support client. Le système traite automatiquement les tickets entrants afin de prédire leur **priorité** (`Low`, `Medium`, `High`, `Critical`) et leur **catégorie de routage** (par exemple `retour_produit`, `facturation`, `support_technique`, `annulation`, `information_produit`) à partir du texte et des métadonnées du ticket [web:30][web:16][web:17].

## Objectif métier

Le projet répond à deux besoins opérationnels :
- **Prioriser** les tickets urgents pour réduire le temps de réponse.
- **Router** les tickets vers la bonne équipe dès leur arrivée [web:423][web:427].

## Variables utilisées

### Texte
- `Ticket Subject`
- `Ticket Description`

### Tabulaire
- `Customer Age`
- `Customer Gender`
- `Product Purchased`
- `Ticket Type`
- `Ticket Channel`

## Sorties attendues

- `priority_label` : priorité du ticket.
- `routing_label` : catégorie de traitement.

Exemple : un ticket de remboursement peut être routé vers `retour_produit`, tandis qu’un ticket technique peut être routé vers `support_technique` [web:423][file:265].

## Structure

```bash
ticket-routing-ready/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## Commandes utiles

```bash
pip install -r requirements.txt
python -m src.data.clean_data
python -m src.models.train_priority --model custom
python -m src.models.train_routing --model custom
python -m src.models.evaluate --task priority --model custom
python -m src.models.evaluate --task routing --model custom
uvicorn src.api.app:app --reload
```
