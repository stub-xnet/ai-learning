# AI_LEARNING — Formation Data Science & Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/ChatGPT-74AA9C?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT">
  <img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/GitHub%20Copilot-000000?style=for-the-badge&logo=githubcopilot&logoColor=white" alt="GitHub Copilot">
  <img src="https://img.shields.io/badge/Cursor-000000?style=for-the-badge&logo=cursor&logoColor=white" alt="Cursor">
  <img src="https://img.shields.io/badge/Mistral%20AI-FA520F?style=for-the-badge&logo=mistralai&logoColor=white" alt="Mistral AI">
  <img src="https://img.shields.io/badge/Perplexity-1FB8CD?style=for-the-badge&logo=perplexity&logoColor=white" alt="Perplexity">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
</p>

Espace de formation personnel : deux cours de référence et trois cahiers de travaux pratiques.

```
AI_LEARNING/
├── README.md                      ← ce fichier
├── requirements.txt               ← bibliothèques (versions testées)
├── .venv/                         ← environnement virtuel dédié (rien d'installé globalement)
│
├── 01_NUMPY_PANDAS/
│   ├── COURS_NUMPY_PANDAS.md      ← cours complet : rappels Python, NumPy (25 chapitres), Pandas (26 chapitres)
│   ├── 01_DEBUTANT.ipynb          ← 40 exercices progressifs
│   ├── 02_INTERMEDIAIRE.ipynb     ← 22 exercices, jeux de données volumineux (e-commerce, capteur, bourse)
│   └── 03_AVANCE.ipynb            ← 17 problèmes complexes + 2 mini-projets complets
│
└── 02_MACHINE_LEARNING/
    └── COURS_MACHINE_LEARNING.md  ← cours complet : de « qu'est-ce que le ML » au déploiement
```

## 1. Démarrer

L'environnement virtuel `.venv` existe déjà. Pour l'utiliser :

```bash
cd ~/AI_LEARNING
source .venv/bin/activate        # à refaire à chaque nouveau terminal
jupyter lab                      # ouvre l'interface (les notebooks utilisent le noyau « Python 3 » du venv)
```

Si `jupyter` ou `xgboost` ne sont pas encore installés dans `.venv` (l'installation de gros paquets peut être interrompue par une connexion instable), complétez-la :

```bash
pip install -r requirements.txt
```

Pour recréer l'environnement de zéro sur une autre machine :

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

> Votre Python global n'a **pas** été modifié : tout est dans `.venv/`.

## 2. Parcours conseillé

| Étape | Contenu | Durée indicative |
|---|---|---|
| 1 | Lire `COURS_NUMPY_PANDAS.md` parties A et B, en tapant les exemples | 1 à 2 semaines |
| 2 | Faire la première moitié de `01_DEBUTANT.ipynb` (sections 1 à 3, NumPy) | 3 à 4 jours |
| 3 | Lire la partie C (Pandas), puis la seconde moitié de `01_DEBUTANT.ipynb` | 1 à 2 semaines |
| 4 | `02_INTERMEDIAIRE.ipynb` | 1 à 2 semaines |
| 5 | `03_AVANCE.ipynb` (problèmes puis mini-projets) | 2 à 3 semaines |
| 6 | `COURS_MACHINE_LEARNING.md` parties A à C (concepts, maths, données) | 1 à 2 semaines |
| 7 | Parties D et E (algorithmes, évaluation), en refaisant chaque exemple sur vos propres données | 3 à 4 semaines |
| 8 | Partie F (projet de bout en bout), puis un projet personnel, puis partie G | 2 semaines et plus |

Ces durées dépendent entièrement de votre rythme : l'essentiel est de **pratiquer** plus que de lire.

## 3. Comment travailler les notebooks

- Chaque exercice a une **consigne** (cellule Markdown) suivie d'une **cellule de code vide** : c'est là que vous répondez.
- Les **corrections** sont regroupées à la fin, dans la section « CORRECTIONS ET EXPLICATIONS ». Chacune contient une solution exécutable, le raisonnement, les erreurs possibles et, souvent, une autre approche.
- Cherchez au moins 15 minutes avant de regarder une correction ; comparez ensuite avec votre propre solution.
- Les jeux de données sont **générés dans le notebook** (graines fixées, donc reproductibles) : exécutez les cellules de préparation avant les exercices (*Run → Run All Cells* fonctionne : vos cellules vides n'affichent rien).
- Faites régulièrement *Kernel → Restart & Run All* pour vérifier que votre notebook fonctionne de haut en bas.

## 4. Ce qui a été vérifié

- Les **162 blocs de code** du cours NumPy/Pandas et les **90 blocs exécutables** du cours Machine Learning ont été exécutés dans l'ordre de lecture, sans erreur.
- Les trois notebooks ont été exécutés **de bout en bout dans un noyau Jupyter** (cellules de réponse vides et corrections comprises), sans erreur. Les solutions contiennent des contrôles (`assert`, comparaison avec une méthode lente) et les résultats affichés ont été relus pour s'assurer qu'ils confirment ce que les explications affirment.
- Versions testées : Python 3.14, NumPy 2.5, Pandas 2.3, scikit-learn 1.9.
- **Non exécutés** (nécessitent des paquets lourds ou optionnels) : les extraits PyTorch, TensorFlow/Keras et FastAPI de la partie G du cours ML, qui sont des illustrations d'introduction.

## 5. Ressources pour la suite

Documentation officielle : <https://numpy.org/doc/>, <https://pandas.pydata.org/docs/>, <https://scikit-learn.org/stable/user_guide.html>. Pour s'entraîner sur des données réelles : Kaggle (Titanic, House Prices, Telco Churn) et UCI Machine Learning Repository.
