# Cours complet de Machine Learning

> **Public visé** : débutant qui maîtrise progressivement Python, NumPy et Pandas (voir `../01_NUMPY_PANDAS/`).
> **Objectif** : comprendre les **logiques** du Machine Learning (pourquoi ça marche, quand ça échoue), pas seulement appeler `model.fit()`.
> **Environnement** : Python ≥ 3.10, NumPy, Pandas, SciPy, scikit-learn ≥ 1.3, Matplotlib (XGBoost pour la partie G). Tous les exemples de code ont été exécutés pour vérification.

## Comment utiliser ce cours

1. Lisez dans l'ordre : chaque partie prépare la suivante (A → mathématiques → données → algorithmes → évaluation → projet).
2. **Exécutez chaque bloc de code** et modifiez les paramètres pour observer ce qui change.
3. Après chaque algorithme, posez-vous trois questions : *Que cherche-t-il à optimiser ? Quelles hypothèses fait-il ? Comment échoue-t-il ?*
4. Faites les exercices de fin de partie.

Encadrés : 💡 **À retenir** · ⚠️ **Piège** · 🎯 **Quand l'utiliser** · 🔍 **Interpréter**.

---

## Table des matières

- [PARTIE A — Comprendre le Machine Learning](#partie-a--comprendre-le-machine-learning)
- [PARTIE B — Les mathématiques nécessaires](#partie-b--les-mathématiques-nécessaires)
- [PARTIE C — Préparation des données](#partie-c--préparation-des-données)
- [PARTIE D — Les algorithmes fondamentaux](#partie-d--les-algorithmes-fondamentaux)
  - D.1 Régression linéaire · D.2 Régression logistique · D.3 K-NN · D.4 Arbres de décision · D.5 Random Forest · D.6 SVM · D.7 Naive Bayes · D.8 K-Means · D.9 Clustering hiérarchique · D.10 ACP · D.11 Tableau comparatif
- [PARTIE E — Évaluation des modèles](#partie-e--évaluation-des-modèles)
- [PARTIE F — Construire un projet de Machine Learning](#partie-f--construire-un-projet-de-machine-learning)
- [PARTIE G — Aller plus loin](#partie-g--aller-plus-loin)
- [Glossaire](#glossaire)
- [Solutions des exercices](#solutions-des-exercices)

---

# PARTIE A — Comprendre le Machine Learning

## A.1 Qu'est-ce que le Machine Learning ?

> **Machine Learning (ML, apprentissage automatique)** : ensemble de méthodes permettant à un programme d'**apprendre des règles à partir d'exemples (données)** plutôt que de recevoir ces règles écrites à la main.

Définition classique (Tom Mitchell, 1997) : un programme **apprend** d'une expérience *E* par rapport à une tâche *T* et une mesure de performance *P* si sa performance sur *T*, mesurée par *P*, s'améliore avec *E*.

Exemple : *T* = reconnaître les courriels indésirables ; *E* = 50 000 courriels étiquetés « spam / non spam » ; *P* = pourcentage de courriels correctement classés.

## A.2 Pourquoi le ML existe-t-il ?

Certains problèmes sont **impossibles à programmer à la main** :

- **Trop de règles** : détecter le spam avec des `if` demanderait des milliers de règles qu'il faut sans cesse mettre à jour.
- **Règles inconnues** : personne ne sait écrire « comment reconnaître un chat sur une photo » en code.
- **Environnement changeant** : les habitudes d'achat évoluent ; un modèle peut être réentraîné.
- **Compréhension des données** : le ML révèle des motifs que l'humain n'avait pas vus.

## A.3 Programmation classique vs Machine Learning

```
Programmation classique :   Données + RÈGLES   ──▶  Programme  ──▶  Résultats
Machine Learning        :   Données + RÉSULTATS ──▶  Apprentissage ──▶  RÈGLES (le modèle)
```

| | Programmation classique | Machine Learning |
|---|---|---|
| Entrées du développeur | règles explicites | exemples (données) |
| Logique | écrite par l'humain | déduite des données |
| Cas nouveaux | échoue si non prévu | généralise (en principe) |
| Maintenance | modifier le code | réentraîner avec de nouvelles données |
| Explicabilité | totale | variable (arbres : bonne ; réseaux : faible) |
| Quand ? | règles claires et stables (calcul d'une facture) | règles floues, complexes ou inconnues (reconnaissance, prédiction) |

## A.4 Comment fonctionne l'apprentissage ?

Boucle fondamentale commune à presque tous les algorithmes supervisés :

1. Choisir une **famille de modèles** paramétrée (ex. une droite `y = a·x + b`, avec paramètres `a`, `b`).
2. Définir une **fonction de coût** (*loss*) mesurant l'erreur du modèle sur les exemples.
3. **Ajuster les paramètres** pour minimiser ce coût (souvent par descente de gradient, partie B).
4. **Évaluer** sur des données jamais vues, pour vérifier que le modèle **généralise**.

```
 données ─▶ modèle(paramètres) ─▶ prédictions ─▶ comparaison aux vraies valeurs ─▶ erreur
                 ▲                                                                   │
                 └──────────── on ajuste les paramètres pour réduire l'erreur ◀──────┘
```

> 💡 **À retenir** : apprendre = **minimiser une erreur** sur des exemples, **en espérant** que cela marche sur des cas inédits. Ce « en espérant » est tout l'enjeu de la partie E (overfitting).

Exemple minimal : prédire le prix d'un appartement.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

surface = np.array([[30], [45], [60], [80], [100]])        # X : (n_obs, n_features)
prix = np.array([45, 62, 80, 105, 130])                    # y : en millions de FCFA

modele = LinearRegression()
modele.fit(surface, prix)                                  # APPRENTISSAGE
print(modele.predict([[70]]))                              # PRÉDICTION → ≈ [92.9]
print(modele.coef_, modele.intercept_)                     # règle apprise : prix ≈ a·surface + b
```

Le programmeur n'a **jamais écrit** « prix ≈ 1,22 × surface + 7,6 » : le modèle l'a déduit des 5 exemples.

## A.5 IA, Machine Learning et Deep Learning

```
┌───────────────────────────────────────────────────────┐
│ Intelligence Artificielle (IA)                        │
│  toute technique qui imite un comportement « intelligent » │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Machine Learning                                │  │
│  │  apprend à partir de données                    │  │
│  │  ┌───────────────────────────────────────────┐  │  │
│  │  │ Deep Learning                             │  │  │
│  │  │  réseaux de neurones profonds             │  │  │
│  │  └───────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

| Terme | Définition | Exemple |
|---|---|---|
| **IA** | cadre général | moteur d'échecs à règles, assistants, voitures autonomes |
| **ML** | l'IA qui apprend des données | filtre de spam, recommandations |
| **Deep Learning** | ML par réseaux de neurones à nombreuses couches | reconnaissance d'images, traduction, modèles de langage |
| **Data Science** | discipline plus large : extraire de la connaissance des données (statistiques + ML + visualisation + métier) | analyse de ventes |

Pour des **données tabulaires** (tableaux Excel), les méthodes classiques (arbres, forêts, boosting) battent souvent le Deep Learning. Le Deep Learning domine pour les **images, sons, textes**.

## A.6 Les types d'apprentissage

| Type | Données | But | Exemples |
|---|---|---|---|
| **Supervisé** | exemples **avec** réponse connue (étiquette) | prédire la réponse pour de nouveaux cas | prix d'une maison, spam, diagnostic |
| **Non supervisé** | exemples **sans** réponse | découvrir une structure | segmentation de clients, réduction de dimension |
| **Par renforcement** | un agent interagit avec un environnement et reçoit des **récompenses** | apprendre une stratégie | jeux (Go, échecs), robots |
| *Semi-supervisé* | peu d'étiquettes + beaucoup de données non étiquetées | profiter des deux | imagerie médicale |
| *Auto-supervisé* | l'étiquette est fabriquée à partir des données | pré-entraîner de gros modèles | modèles de langage |

### Apprentissage supervisé (*Supervised Learning*)

On dispose de paires `(entrée, sortie attendue)`. Le modèle apprend la correspondance. Deux sous-familles :

| | Régression | Classification |
|---|---|---|
| Sortie | **valeur numérique continue** | **catégorie (classe)** |
| Exemples | prix, température, durée, ventes | spam/non spam, espèce, fraude oui/non |
| Mesures | MAE, RMSE, R² | accuracy, précision, rappel, F1 |
| Algorithmes | régression linéaire, forêt (regressor) | régression logistique, K-NN, SVM… |

⚠️ **Piège** : la « régression logistique » est un algorithme de **classification**, malgré son nom.

La classification peut être **binaire** (2 classes), **multi-classes** (3+, une seule vraie), ou **multi-étiquettes** (plusieurs vraies simultanément).

### Apprentissage non supervisé (*Unsupervised Learning*)

Pas d'étiquette. Deux grandes tâches :

- **Clustering** (regroupement) : K-Means, clustering hiérarchique, DBSCAN.
- **Réduction de dimension** : ACP/PCA, t-SNE, UMAP.
- Aussi : détection d'anomalies, règles d'association.

### Apprentissage par renforcement (*Reinforcement Learning*)

Un **agent** observe l'**état** d'un environnement, choisit une **action**, reçoit une **récompense** et un nouvel état. Il apprend une **politique** maximisant la récompense cumulée. Hors du périmètre de ce cours (voir partie G).

## A.7 Vocabulaire fondamental

| Terme | Sens | Équivalent Pandas/NumPy |
|---|---|---|
| **Dataset** | ensemble des données | `DataFrame` |
| **Observation / échantillon / instance** | une ligne = un exemple | une ligne |
| **Feature / variable / caractéristique / attribut** | une colonne d'entrée | une colonne de `X` |
| **Label / cible / target** | la réponse à prédire | le vecteur `y` |
| **X** | matrice des features, forme `(n_observations, n_features)` | `df.drop(columns="cible")` |
| **y** | vecteur des labels, forme `(n_observations,)` | `df["cible"]` |
| **Paramètres** | valeurs **apprises** (coefficients, seuils) | `model.coef_` |
| **Hyperparamètres** | réglages **fixés avant** l'apprentissage (`k` de K-NN, profondeur d'un arbre) | arguments du constructeur |
| **Entraînement (`fit`)** | ajustement des paramètres | `model.fit(X, y)` |
| **Prédiction (`predict`)** | production des sorties pour de nouvelles entrées | `model.predict(X_new)` |

Exemple : prédire si un client va se désabonner.

| ancienneté (mois) | factures impayées | forfait | **résilié** |
|---|---|---|---|
| 24 | 0 | premium | non |
| 3 | 2 | basique | oui |
| 12 | 1 | basique | non |

- 3 **observations**, 3 **features** (ancienneté, impayés, forfait), 1 **label** (résilié).
- `X` est de forme `(3, 3)`, `y` de forme `(3,)`.

```python
import pandas as pd
df = pd.DataFrame({"anciennete": [24, 3, 12], "impayes": [0, 2, 1],
                   "forfait": ["premium", "basique", "basique"], "resilie": ["non", "oui", "non"]})
X = df.drop(columns="resilie")      # features
y = df["resilie"]                   # label
print(X.shape, y.shape)             # (3, 3) (3,)
```

## A.8 Les trois temps d'un modèle

1. **Entraînement** : `model.fit(X_train, y_train)`.
2. **Évaluation** : `model.score(X_test, y_test)` sur des données **non vues**.
3. **Prédiction** (utilisation) : `model.predict(X_nouveau)`.

> 💡 La règle d'or : **on n'évalue jamais un modèle sur les données qui ont servi à l'entraîner.**

## Exercices A

1. Pour chaque cas, dites s'il s'agit de régression, classification, clustering ou autre : (a) prévoir la température de demain ; (b) détecter une transaction frauduleuse ; (c) regrouper des clients par comportement d'achat ; (d) estimer la durée d'un trajet ; (e) reconnaître le chiffre manuscrit sur un chèque.
2. Dans un tableau de 1 000 étudiants avec 8 colonnes dont « admis (oui/non) », que valent `n_observations`, `n_features`, la forme de `X` et celle de `y` ?
3. Pourquoi un paramètre et un hyperparamètre sont-ils différents ? Donnez un exemple de chaque pour un arbre de décision.
4. Pourquoi la programmation classique est-elle meilleure pour calculer une TVA ?

---

# PARTIE B — Les mathématiques nécessaires

Les mathématiques ne sont pas un obstacle : on a besoin d'**intuitions** plus que de démonstrations. Pour chaque notion, nous indiquons son **utilité réelle** en ML.

## B.1 Algèbre linéaire

### B.1.1 Vecteurs

Un **vecteur** est une liste ordonnée de nombres. Géométriquement : une flèche ou un point dans l'espace. En ML : **une observation est un vecteur** de ses features.

```python
import numpy as np
x = np.array([170, 65, 21])      # taille (cm), poids (kg), âge : un vecteur à 3 composantes
```

Opérations :

| Opération | Formule | NumPy | Sens |
|---|---|---|---|
| Addition | `u + v` composante par composante | `u + v` | combiner |
| Multiplication par un scalaire | `λ·u` | `2 * u` | étirer |
| **Produit scalaire** | `u·v = Σ uᵢvᵢ` | `u @ v` | mesure d'alignement/similarité |
| **Norme** | `‖u‖ = √(Σ uᵢ²)` | `np.linalg.norm(u)` | longueur |
| **Distance** | `‖u − v‖` | `np.linalg.norm(u - v)` | éloignement de deux observations |

```python
u = np.array([3, 4]); v = np.array([1, 2])
print(u @ v)                           # 3·1 + 4·2 = 11
print(np.linalg.norm(u))               # 5.0
print(np.linalg.norm(u - v))           # √(4+4) = 2.828...
cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print(round(cos, 3))                   # 0.984 : très proches en direction
```

🎯 **Utilité ML** : la distance mesure la ressemblance (K-NN, K-Means) ; le produit scalaire est l'opération de base d'un modèle linéaire (`prédiction = w · x + b`) et d'un neurone.

### B.1.2 Matrices

Une **matrice** est un tableau 2D de nombres. En ML, le dataset complet `X` est une matrice `(n_observations × n_features)`.

```python
X = np.array([[170, 65], [160, 55], [180, 80]])    # 3 personnes × 2 features
print(X.shape)       # (3, 2)
print(X.T)           # transposée (2, 3)
```

**Produit matriciel** `A @ B` : `(m, n) @ (n, p) → (m, p)`. Chaque case `(i, j)` du résultat est le produit scalaire de la ligne `i` de `A` par la colonne `j` de `B`.

```python
w = np.array([0.5, 2.0])
print(X @ w)         # prédiction linéaire pour les 3 personnes : [215. 190. 250.]
```

Autres notions :

| Notion | Définition | Utilité ML |
|---|---|---|
| **Matrice identité `I`** | 1 sur la diagonale, 0 ailleurs ; `A @ I = A` | neutre |
| **Inverse `A⁻¹`** | `A @ A⁻¹ = I` (si elle existe) | équation normale de la régression |
| **Déterminant** | nul ⇒ non inversible | colinéarité entre variables |
| **Valeurs/vecteurs propres** | `A v = λ v` : directions qui ne sont qu'étirées | **ACP** |
| **Rang** | nb de colonnes indépendantes | variables redondantes |

**Valeurs propres : intuition.** Une matrice transforme l'espace (rotation, étirement). Ses vecteurs propres sont les directions qui ne tournent pas, juste étirées par le facteur λ. Pour une matrice de covariance, ces directions sont celles où les données varient le plus : c'est le cœur de l'ACP (D.10).

```python
C = np.array([[2.0, 1.0], [1.0, 2.0]])
valeurs, vecteurs = np.linalg.eigh(C)
print(valeurs)            # [1. 3.]
```

### B.1.3 Pourquoi tout cela ?

Une régression linéaire à 3 features et 1000 observations se résume à une opération : `y_pred = X @ w + b`. L'algèbre linéaire permet d'écrire — et de calculer **très vite** — des milliers de calculs d'un coup.

## B.2 Statistiques descriptives

Décrire un jeu de données avec quelques nombres.

| Mesure | Formule | Lecture |
|---|---|---|
| **Moyenne** `μ` | `(1/n) Σ xᵢ` | centre, sensible aux extrêmes |
| **Médiane** | valeur centrale | centre, robuste |
| **Mode** | valeur la plus fréquente | catégories |
| **Variance** `σ²` | `(1/n) Σ (xᵢ − μ)²` | dispersion (unité au carré) |
| **Écart-type** `σ` | `√variance` | dispersion typique, même unité que `x` |
| **Quartiles / IQR** | Q1, Q3, `IQR = Q3 − Q1` | dispersion robuste, détection d'outliers |
| **Étendue** | `max − min` | amplitude |

```python
import numpy as np
x = np.array([12, 15, 9, 18, 14, 11, 16])
print(x.mean(), np.median(x), x.var(), x.std())
```

> 💡 **Variance de l'échantillon** : pour estimer la variance d'une population à partir d'un échantillon, on divise par `n − 1` (correction de Bessel, `ddof=1`) ; sinon on la sous-estime.

🎯 **Utilité ML** : repérer les échelles très différentes des variables (→ standardisation, partie C), les outliers, les distributions asymétriques.

Histogramme mental : la moyenne et l'écart-type résument bien une distribution en forme de cloche ; pour une distribution très asymétrique (revenus), préférez médiane et quantiles.

## B.3 Probabilités

Une **probabilité** `P(A)` ∈ [0, 1] quantifie l'incertitude d'un événement `A`.

| Notion | Formule | Exemple |
|---|---|---|
| Événement complémentaire | `P(non A) = 1 − P(A)` | |
| Union | `P(A ou B) = P(A) + P(B) − P(A et B)` | |
| Indépendance | `P(A et B) = P(A)·P(B)` | deux lancers de pièce |
| **Probabilité conditionnelle** | `P(A│B) = P(A et B) / P(B)` | proba de pluie sachant qu'il y a des nuages |
| **Théorème de Bayes** | `P(A│B) = P(B│A)·P(A) / P(B)` | inverser un conditionnel |

### Théorème de Bayes : un exemple qui surprend

Une maladie touche 1 % de la population. Un test détecte 95 % des malades (sensibilité) et donne 5 % de faux positifs. Vous êtes positif : quelle probabilité d'être malade ?

```python
p_m = 0.01
p_pos_si_m = 0.95
p_pos_si_non_m = 0.05
p_pos = p_pos_si_m * p_m + p_pos_si_non_m * (1 - p_m)
p_m_si_pos = p_pos_si_m * p_m / p_pos
print(round(p_m_si_pos, 3))        # 0.161 → seulement 16 %, pas 95 % !
```

Explication : sur 10 000 personnes, 100 sont malades (95 positifs) et 9 900 saines (495 faux positifs). Sur 590 positifs, seuls 95 sont malades : 95/590 ≈ 16 %. **La rareté de la maladie domine.**

🎯 **Utilité ML** : le classifieur **Naive Bayes** applique directement ce théorème ; la sortie de la régression logistique est une probabilité ; les métriques *précision* et *rappel* (partie E) relèvent de ce raisonnement (déséquilibre de classes).

## B.4 Distributions

Une **distribution** décrit comment les valeurs d'une variable aléatoire se répartissent.

| Loi | Forme | Paramètres | Exemple | NumPy |
|---|---|---|---|---|
| **Normale (gaussienne)** | cloche symétrique | moyenne `μ`, écart-type `σ` | tailles, erreurs de mesure | `rng.normal(mu, sigma, n)` |
| **Uniforme** | plate | bornes | tirage équiprobable | `rng.uniform(a, b, n)` |
| **Bernoulli** | 0/1 | `p` | un client achète (oui/non) | `rng.binomial(1, p, n)` |
| **Binomiale** | nb de succès sur `n` essais | `n`, `p` | nb de clics sur 100 vues | `rng.binomial(n, p, size)` |
| **Poisson** | comptage d'événements rares | `λ` | appels par heure | `rng.poisson(lam, n)` |
| **Exponentielle** | temps d'attente | taux | durée entre deux pannes | `rng.exponential(scale, n)` |

Règle **68-95-99,7** pour la loi normale : environ 68 % des valeurs sont à ±1σ de la moyenne, 95 % à ±2σ, 99,7 % à ±3σ.

```python
rng = np.random.default_rng(0)
x = rng.normal(loc=170, scale=10, size=100_000)
print(((x > 160) & (x < 180)).mean().round(3))     # ≈ 0.683
print(((x > 150) & (x < 190)).mean().round(3))     # ≈ 0.954
```

**Théorème central limite** : la moyenne de nombreuses observations indépendantes tend vers une loi normale, quelle que soit la distribution de départ. C'est pourquoi la loi normale est partout.

```python
moyennes = rng.exponential(scale=2.0, size=(10_000, 50)).mean(axis=1)
print(moyennes.mean().round(2), moyennes.std().round(2))     # ≈ 2.0, ≈ 0.28 (= 2/√50)
```

🎯 **Utilité ML** : beaucoup de modèles (régression linéaire, Naive Bayes gaussien, ACP) font des hypothèses sur la distribution ; la standardisation suppose une échelle « raisonnable » ; on initialise les poids des réseaux avec des tirages normaux.

Statistiques utiles : **asymétrie** (`df.skew()`), possibilité de transformation `np.log1p` pour les distributions à longue queue (revenus, prix).

## B.5 Covariance et corrélation

Comment deux variables varient-elles **ensemble** ?

**Covariance** : `cov(X, Y) = (1/(n−1)) Σ (xᵢ − x̄)(yᵢ − ȳ)`.
- > 0 : elles montent ensemble ; < 0 : l'une monte quand l'autre descend ; ≈ 0 : pas de lien linéaire.
- Dépend des unités, donc difficile à interpréter.

**Corrélation de Pearson** : covariance normalisée par les écarts-types, toujours entre −1 et 1.

`r = cov(X, Y) / (σ_X · σ_Y)`

| `r` | Interprétation |
|---|---|
| ≈ +1 | forte relation linéaire croissante |
| ≈ 0 | pas de relation **linéaire** |
| ≈ −1 | forte relation linéaire décroissante |

```python
import numpy as np
heures = np.array([1, 2, 3, 4, 5, 6, 7, 8])
note = np.array([8, 9, 11, 12, 13, 15, 16, 18])
print(np.cov(heures, note)[0, 1].round(2))       # covariance
print(np.corrcoef(heures, note)[0, 1].round(3))  # ≈ 0.99
```

⚠️ **Pièges**
1. **Corrélation ≠ causalité** : les ventes de glaces et les noyades sont corrélées (l'été les explique toutes deux).
2. **`r ≈ 0` n'implique pas indépendance** : `y = x²` sur un intervalle symétrique donne `r ≈ 0` alors que la dépendance est totale.

```python
x = np.linspace(-3, 3, 101)
print(np.corrcoef(x, x ** 2)[0, 1].round(3))    # ≈ 0 pourtant y = x²
```

3. Sensible aux outliers (voir aussi la corrélation de Spearman, basée sur les rangs : `df.corr(method="spearman")`).

**Matrice de covariance / de corrélation** pour plusieurs variables : `df.corr()`, `np.cov(X, rowvar=False)`.

🎯 **Utilité ML** : repérer les features redondantes (corrélation > 0,9 → multicolinéarité), sélectionner celles liées à la cible, comprendre l'ACP (elle diagonalise la matrice de covariance).

## B.6 Dérivées

La **dérivée** `f'(x)` mesure la **vitesse de variation** de `f` en `x` : de combien `f` change quand `x` augmente un tout petit peu. Géométriquement : la **pente de la tangente**.

| Fonction | Dérivée |
|---|---|
| `f(x) = c` | `0` |
| `f(x) = a·x + b` | `a` |
| `f(x) = x²` | `2x` |
| `f(x) = xⁿ` | `n·xⁿ⁻¹` |
| `f(x) = eˣ` | `eˣ` |
| `f(x) = ln x` | `1/x` |

Règles : `(f+g)' = f'+g'` ; `(c·f)' = c·f'` ; **règle de la chaîne** `(f(g(x)))' = f'(g(x))·g'(x)` (fondamentale pour la rétropropagation dans les réseaux de neurones).

**Signification pour optimiser** :
- `f'(x) > 0` : `f` croît ⇒ pour la **diminuer**, on recule.
- `f'(x) < 0` : `f` décroît ⇒ on avance.
- `f'(x) = 0` : point stationnaire (minimum, maximum ou col).

Vérification numérique :

```python
f = lambda x: x ** 2
x0, h = 3.0, 1e-6
print((f(x0 + h) - f(x0 - h)) / (2 * h))     # ≈ 6.0 = 2·3
```

🎯 **Utilité ML** : pour trouver les paramètres qui minimisent l'erreur, on cherche l'endroit où la **pente de la fonction de coût** s'annule.

## B.7 Gradient

Quand la fonction dépend de **plusieurs** variables `f(w₁, w₂, …)`, on utilise les **dérivées partielles** (on dérive par rapport à une variable, les autres figées). Le **gradient** `∇f` est le vecteur de toutes les dérivées partielles :

`∇f = (∂f/∂w₁, ∂f/∂w₂, …)`

Propriété clé : le gradient pointe vers la **direction de plus forte augmentation** de `f`. Donc `−∇f` pointe vers la plus forte **diminution**.

Exemple : `f(w₁, w₂) = w₁² + 3·w₂²` ⇒ `∇f = (2w₁, 6w₂)`. En `(1, 1)`, `∇f = (2, 6)` : `w₂` pèse davantage.

```python
def grad_f(w):
    return np.array([2 * w[0], 6 * w[1]])
print(grad_f(np.array([1.0, 1.0])))      # [2. 6.]
```

## B.8 Descente de gradient

**Idée** : pour minimiser une fonction de coût `J(w)`, on part d'un point, on calcule le gradient, et on fait un petit pas dans la direction opposée. On répète jusqu'à convergence.

`w ← w − η · ∇J(w)`

où **`η` (eta)** est le **taux d'apprentissage** (*learning rate*) : la taille du pas.

Image : on descend une montagne dans le brouillard en suivant la pente locale la plus forte vers le bas.

### Étude complète : régression linéaire simple par descente de gradient

Modèle : `ŷ = w·x + b`. Coût : erreur quadratique moyenne `J(w, b) = (1/n) Σ (ŷᵢ − yᵢ)²`.

Gradients :
- `∂J/∂w = (2/n) Σ (ŷᵢ − yᵢ)·xᵢ`
- `∂J/∂b = (2/n) Σ (ŷᵢ − yᵢ)`

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 100)
y = 3.0 * x + 5.0 + rng.normal(0, 2, 100)          # vérité : w=3, b=5, + bruit

w, b = 0.0, 0.0
eta = 0.01
historique = []
for epoch in range(2000):
    y_pred = w * x + b
    erreur = y_pred - y
    grad_w = 2 * (erreur * x).mean()
    grad_b = 2 * erreur.mean()
    w -= eta * grad_w
    b -= eta * grad_b
    historique.append((erreur ** 2).mean())

print(round(w, 2), round(b, 2))      # ≈ 3.0, ≈ 5.0
print(round(historique[0], 1), round(historique[-1], 2))   # coût qui chute (≈ 4)
```

### Influence du taux d'apprentissage

| `η` | Comportement |
|---|---|
| trop petit | convergence très lente |
| bien choisi | convergence régulière |
| trop grand | oscille, voire **diverge** (le coût explose) |

```python
for eta in [0.0001, 0.01, 0.05]:
    w, b = 0.0, 0.0
    for _ in range(200):
        e = w * x + b - y
        w -= eta * 2 * (e * x).mean()
        b -= eta * 2 * e.mean()
    print(eta, round(((w * x + b - y) ** 2).mean(), 2))
# 0.0001 → coût encore élevé ; 0.01 → proche de l'optimum ; 0.05 → divergence (nan/inf ou immense)
```

### Variantes

| Variante | Données utilisées par pas | Caractéristique |
|---|---|---|
| **Batch** | tout le dataset | stable, lent sur gros volumes |
| **Stochastique (SGD)** | 1 observation | rapide, bruité |
| **Mini-batch** | petit lot (32–256) | compromis, standard en Deep Learning |

Extensions : momentum, RMSprop, **Adam** (le plus utilisé).

⚠️ **Pièges** : les features à échelles très différentes rendent la descente zigzagante (d'où la **standardisation**, partie C) ; certains coûts ont des minima locaux (réseaux de neurones) mais pas la régression linéaire ni logistique (coût convexe).

🎯 **Utilité ML** : c'est le moteur d'entraînement de la régression logistique, des réseaux de neurones, du gradient boosting (descente de gradient dans l'espace des fonctions).

## B.9 Résumé : mathématiques ↔ ML

| Concept | Où il apparaît |
|---|---|
| Vecteur / norme / distance | K-NN, K-Means, SVM |
| Produit matriciel | prédiction linéaire, réseaux de neurones |
| Valeurs propres | ACP |
| Moyenne/variance | standardisation, ACP, Naive Bayes |
| Probabilités / Bayes | Naive Bayes, régression logistique, incertitude |
| Distribution normale | hypothèses de modèles, initialisation |
| Corrélation | sélection de variables, diagnostic |
| Dérivée / gradient | entraînement par descente de gradient |

## Exercices B

1. Calculez à la main puis avec NumPy la distance entre `A=(1,2)` et `B=(4,6)`.
2. Pour `X` de forme `(100, 5)` et `w` de forme `(5,)`, quelle est la forme de `X @ w` ? Pourquoi `w @ X` échoue-t-il ?
3. Dans un test médical, sensibilité 90 %, taux de faux positifs 10 %, prévalence 2 %. Calculez la probabilité d'être malade si positif.
4. Pour `f(x) = x² − 4x + 5`, trouvez à la main le minimum (dérivée nulle), puis vérifiez avec 50 pas de descente de gradient avec `η = 0.1` partant de `x = 0`.
5. Générez `x` normal et `y = 2x + bruit`. Calculez la corrélation. Que devient-elle si vous remplacez `y` par `y = x²` ? Expliquez.

---

# PARTIE C — Préparation des données

> **« Garbage in, garbage out »** : la qualité des données détermine la qualité du modèle. Dans un projet réel, la préparation représente souvent 60 à 80 % du travail.

Le jeu de données utilisé dans cette partie est généré pour être **réaliste** : prêts bancaires fictifs, avec valeurs manquantes, outliers et variables catégorielles.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000
age = rng.integers(21, 66, n)
revenu = np.round(rng.lognormal(mean=np.log(250_000), sigma=0.45, size=n), -2)      # FCFA/mois
anciennete = np.clip(age - 21 - rng.integers(0, 10, n), 0, None)
niveau = rng.choice(["Aucun", "Bac", "Licence", "Master"], n, p=[0.1, 0.3, 0.4, 0.2])
statut = rng.choice(["Salarié", "Indépendant", "Fonctionnaire", "Sans emploi"], n, p=[0.45, 0.25, 0.2, 0.1])
montant = np.round(revenu * rng.uniform(2, 12, n), -3)
dettes = np.round(revenu * rng.uniform(0, 2.5, n), -2)

# La probabilité de défaut dépend réellement des variables : le jeu est cohérent
score = (-4.4 + 1.6 * (dettes / revenu) + 0.12 * (montant / revenu) - 0.015 * anciennete
         + (statut == "Sans emploi") * 1.2 + (statut == "Indépendant") * 0.3
         - (niveau == "Master") * 0.4 + rng.normal(0, 0.5, n))
defaut = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)

credit = pd.DataFrame({"age": age, "revenu": revenu, "anciennete": anciennete, "niveau": niveau,
                       "statut": statut, "montant": montant, "dettes": dettes, "defaut": defaut})

# Imperfections réalistes
credit.loc[rng.choice(n, 60, replace=False), "revenu"] = np.nan
credit.loc[rng.choice(n, 40, replace=False), "niveau"] = np.nan
credit.loc[rng.choice(n, 5, replace=False), "revenu"] *= 40          # outliers (erreurs de saisie)
credit = pd.concat([credit, credit.sample(8, random_state=1)], ignore_index=True)      # doublons
print(credit.shape, credit["defaut"].mean().round(3))
```

## C.1 Collecte des données

Sources : fichiers (CSV, Excel), bases de données (SQL), API web, capteurs, web scraping (dans le respect du droit et des conditions d'usage), jeux publics (UCI, Kaggle, data.gouv).

Questions à se poser **avant** de modéliser :

1. **Représentativité** : les données reflètent-elles la situation où le modèle sera utilisé ?
2. **Taille** : assez d'exemples pour le nombre de variables ?
3. **Biais** : certaines populations sont-elles absentes ou sous-représentées ?
4. **Fuite temporelle** : sait-on au moment de la prédiction ce qu'on met en feature ?
5. **Éthique et légalité** : données personnelles, consentement (RGPD, lois locales).
6. **Qualité de l'étiquette** : le label est-il fiable ?

## C.2 Exploration des datasets (EDA)

L'**analyse exploratoire** (*Exploratory Data Analysis*) précède toute modélisation.

```python
print(credit.shape)
print(credit.dtypes)
print(credit.describe().round(0))
print(credit.isna().sum())
print(credit.duplicated().sum())
print(credit["defaut"].value_counts(normalize=True))            # équilibre des classes
print(credit.groupby("statut")["defaut"].mean().round(3))       # lien variable ↔ cible
print(credit.select_dtypes("number").corr()["defaut"].round(2).sort_values())
```

À regarder systématiquement :

| Aspect | Outil | Ce qu'on cherche |
|---|---|---|
| Structure | `shape`, `info()` | types, volume |
| Distributions | `hist`, `describe()` | asymétrie, outliers |
| Cible | `value_counts(normalize=True)` | déséquilibre de classes |
| Manquants | `isna().mean()` | colonnes inutilisables |
| Doublons | `duplicated().sum()` | répétitions |
| Relations | `corr()`, `groupby` | variables informatives |
| Incohérences | règles métier | âge négatif, date future |

Visualisation avec Matplotlib :

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
credit["revenu"].hist(bins=40, ax=axes[0]); axes[0].set_title("Revenu")
credit.boxplot(column="revenu", ax=axes[1]);  axes[1].set_title("Boîte à moustaches")
credit.groupby("statut")["defaut"].mean().plot.bar(ax=axes[2], title="Taux de défaut par statut")
plt.tight_layout(); plt.show()
```

## C.3 Nettoyage

```python
propre = credit.drop_duplicates().copy()           # 1. doublons
print(len(credit), "→", len(propre))

# 2. Outliers : règle de l'IQR (sur la colonne revenu)
q1, q3 = propre["revenu"].quantile([0.25, 0.75])
iqr = q3 - q1
borne_haute = q3 + 3 * iqr                         # 3×IQR : seuil « extrême »
print((propre["revenu"] > borne_haute).sum(), "valeurs extrêmes")
propre.loc[propre["revenu"] > borne_haute, "revenu"] = np.nan     # traités comme manquants
```

Traitements possibles des outliers : conserver (si légitimes), corriger (erreur de saisie), **plafonner** (`clip`, *winsorization*), transformer (`log`), ou supprimer (en dernier recours). Leur traitement dépend de la **cause** : erreur ou phénomène réel ?

## C.4 Valeurs manquantes

Trois mécanismes (théorie de Rubin) :

| Mécanisme | Signification | Exemple |
|---|---|---|
| **MCAR** | manque complètement au hasard | capteur tombé en panne aléatoirement |
| **MAR** | manque lié à *d'autres* variables observées | les jeunes renseignent moins leur revenu |
| **MNAR** | manque lié à la valeur elle-même | les hauts revenus cachent leur revenu |

Stratégies :

| Méthode | Description | Avantage / limite |
|---|---|---|
| Suppression de lignes | `dropna()` | simple ; perte de données/biais |
| Suppression de colonne | si > 60 % de NA | perte d'information |
| Moyenne / médiane | `SimpleImputer(strategy="median")` | rapide ; réduit la variance |
| Mode / constante | catégories | simple |
| Indicateur de manque | colonne `revenu_manquant` | le manque est parfois informatif |
| K-NN Imputer | moyenne des voisins | respecte les relations ; plus lent |
| Imputation itérative | modèle prédictif par colonne (`IterativeImputer`) | précis ; complexe |

```python
from sklearn.impute import SimpleImputer, KNNImputer

imp_num = SimpleImputer(strategy="median")
propre[["revenu"]] = imp_num.fit_transform(propre[["revenu"]])
imp_cat = SimpleImputer(strategy="most_frequent")
propre[["niveau"]] = imp_cat.fit_transform(propre[["niveau"]])
print(propre.isna().sum().sum())      # 0
```

⚠️ **Piège majeur** : calculer la médiane sur **tout** le dataset avant le split train/test est une *fuite de données* (C.9). La bonne pratique est d'**ajuster (`fit`) l'imputeur uniquement sur le train** (voir `Pipeline`, F.6).

## C.5 Encodage des variables catégorielles

Les algorithmes manipulent des nombres : il faut convertir les catégories.

| Type | Exemple | Encodage recommandé |
|---|---|---|
| **Nominale** (pas d'ordre) | statut, ville, couleur | **One-Hot** |
| **Ordinale** (ordre) | niveau d'études, taille S/M/L | **Ordinal** (ordre explicite) |
| **Cible binaire** | oui/non | `LabelEncoder` / 0-1 |
| **Haute cardinalité** (1000 villes) | code postal | encodage par fréquence ou par cible (avec précautions) |

### One-Hot Encoding

Une colonne binaire par modalité.

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
statut_oh = enc.fit_transform(propre[["statut"]])
print(enc.get_feature_names_out())
# ['statut_Fonctionnaire' 'statut_Indépendant' 'statut_Salarié' 'statut_Sans emploi']
print(statut_oh[:3])
```

Équivalent Pandas : `pd.get_dummies(df, columns=["statut"], drop_first=True)`.

⚠️ **Piège de la « multicolinéarité »** : avec `k` modalités, `k` colonnes sont redondantes (leur somme vaut 1) : pour la régression linéaire, utilisez `drop="first"`.

### Encodage ordinal

```python
from sklearn.preprocessing import OrdinalEncoder
ordre = [["Aucun", "Bac", "Licence", "Master"]]
oe = OrdinalEncoder(categories=ordre)
propre["niveau_num"] = oe.fit_transform(propre[["niveau"]]).ravel()
print(propre["niveau_num"].value_counts().sort_index())
```

⚠️ **Piège** : encoder `Cotonou=0, Parakou=1, Porto-Novo=2` avec des entiers suggère un ordre et des distances inexistantes (Porto-Novo « = 2× » Parakou ?). Réservez l'encodage ordinal aux variables vraiment ordonnées.

## C.6 Normalisation et standardisation (mise à l'échelle)

Les algorithmes fondés sur des **distances** ou des **gradients** (K-NN, K-Means, SVM, régression logistique, réseaux de neurones, ACP) sont sensibles aux échelles : un revenu en dizaines de milliers écrase un âge de 20 à 60.

| Méthode | Formule | Résultat | Quand ? |
|---|---|---|---|
| **Standardisation** (z-score) `StandardScaler` | `(x − μ) / σ` | moyenne 0, écart-type 1 | par défaut ; sensible aux outliers |
| **Normalisation min-max** `MinMaxScaler` | `(x − min) / (max − min)` | entre 0 et 1 | bornes connues, réseaux de neurones, images |
| **Robust** `RobustScaler` | `(x − médiane) / IQR` | robuste aux outliers | données avec valeurs extrêmes |
| **Log** | `np.log1p(x)` | réduit l'asymétrie | variables à longue queue |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
cols = ["age", "revenu", "montant"]
print(propre[cols].describe().loc[["mean", "std"]].round(0))

sc = StandardScaler().fit(propre[cols])
Z = sc.transform(propre[cols])
print(Z.mean(axis=0).round(6), Z.std(axis=0).round(3))     # [0 0 0] [1 1 1]
```

**Qui n'en a pas besoin ?** Les arbres de décision, forêts aléatoires et gradient boosting (ils comparent des seuils, insensibles aux échelles monotones) ; Naive Bayes.

⚠️ **Règle** : `fit` du scaler **sur le train seulement**, puis `transform` sur train **et** test avec les mêmes paramètres (μ, σ du train).

## C.7 Feature Engineering

Créer de nouvelles variables plus informatives à partir des existantes — c'est souvent là que se gagne la performance.

```python
propre["ratio_dette_revenu"] = propre["dettes"] / propre["revenu"]
propre["ratio_montant_revenu"] = propre["montant"] / propre["revenu"]
propre["log_revenu"] = np.log1p(propre["revenu"])
propre["revenu_manquant"] = credit.drop_duplicates()["revenu"].isna().astype(int).values
propre["tranche_age"] = pd.cut(propre["age"], [0, 30, 45, 60, 100], labels=["<30", "30-45", "45-60", "60+"])
```

Techniques courantes :

| Technique | Exemple |
|---|---|
| Ratios | dette/revenu, prix/m² |
| Transformations | `log`, racine carrée, carré |
| Interactions | `âge × revenu` ; `PolynomialFeatures` |
| Discrétisation | tranches d'âge |
| Dates | jour de semaine, mois, week-end, ancienneté |
| Texte | longueur, présence de mots, TF-IDF |
| Agrégats | moyenne des achats par client (`groupby`) |
| Indicateurs | « valeur manquante », « week-end » |

> 💡 Un expert du domaine suggère souvent la meilleure variable : le ratio dette/revenu est plus informatif que la dette seule.

## C.8 Sélection des variables

Moins de variables = modèle plus simple, plus rapide, moins sujet à l'overfitting, plus interprétable.

| Famille | Principe | Outils |
|---|---|---|
| **Filtres** | score statistique indépendant du modèle | variance faible (`VarianceThreshold`), corrélation, `SelectKBest(f_classif)` |
| **Wrappers** | essayer des sous-ensembles avec un modèle | `RFE` (élimination récursive) |
| **Embarquées** | la sélection fait partie de l'apprentissage | Lasso (L1), importance des arbres |

```python
from sklearn.feature_selection import SelectKBest, f_classif
num = propre[["age", "revenu", "anciennete", "montant", "dettes", "ratio_dette_revenu", "ratio_montant_revenu"]]
sel = SelectKBest(f_classif, k=3).fit(num, propre["defaut"])
print(num.columns[sel.get_support()].tolist())
```

Élimination des redondances : si deux features ont une corrélation > 0,9, n'en gardez qu'une.

⚠️ La sélection fondée sur la cible doit aussi se faire **à l'intérieur de la validation croisée** (fuite sinon).

## C.9 Séparation Train / Test

On réserve une partie des données que le modèle **ne verra jamais pendant l'entraînement** : elle simule les données futures.

```python
from sklearn.model_selection import train_test_split

X = propre[["age", "revenu", "anciennete", "montant", "dettes", "statut", "niveau"]]
y = propre["defaut"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)
print(y_train.mean().round(3), y_test.mean().round(3))      # proportions de défauts quasi identiques
```

| Paramètre | Rôle |
|---|---|
| `test_size` | part du test (0,2 à 0,3 courant) |
| `random_state` | reproductibilité |
| `stratify=y` | conserve les proportions des classes (**indispensable** si classes déséquilibrées) |
| `shuffle=False` | pour les séries temporelles (couper chronologiquement) |

Schéma complet à trois jeux (pour régler des hyperparamètres) :

```
Données ──▶ Train (60-70 %) : apprendre les paramètres
        ├─▶ Validation (15-20 %) : choisir les hyperparamètres / comparer les modèles
        └─▶ Test (15-20 %) : évaluation FINALE, utilisée UNE fois
```

La validation croisée (partie E) remplace avantageusement un jeu de validation fixe.

## C.10 Data Leakage (fuite de données)

> **Data leakage** : de l'information issue du test (ou de l'avenir) pénètre dans l'entraînement. Les résultats semblent excellents mais **s'effondrent en production**.

### Types de fuites

| Type | Exemple | Remède |
|---|---|---|
| **Prétraitement avant split** | `scaler.fit(X_complet)` puis split | split d'abord ; `Pipeline` |
| **Feature issue de la cible** | prédire un défaut de paiement avec la colonne « nombre de relances » (créée après le défaut) | n'utiliser que ce qui est connu **au moment de la prédiction** |
| **Fuite temporelle** | mélanger passé/futur dans une série | split chronologique |
| **Doublons train/test** | la même observation des deux côtés | dédoublonner avant |
| **Sélection de features sur tout le dataset** | `SelectKBest` avant split | dans le pipeline |
| **Groupes** | plusieurs lignes du même patient réparties train/test | `GroupKFold` |

### Démonstration chiffrée

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2000))          # 2000 variables PUREMENT aléatoires
y = rng.integers(0, 2, 100)               # cible aléatoire : AUCUN lien possible

# ✘ Fuite : on sélectionne les 20 « meilleures » variables sur TOUT le dataset
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
print("avec fuite    :", cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean().round(2))   # ≈ 0.75-0.8 !

# ✔ Sans fuite : la sélection est refaite dans chaque pli
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
print("sans fuite    :", cross_val_score(pipe, X, y, cv=5).mean().round(2))                       # ≈ 0.5-0.6 (hasard)
```

La première évaluation prétend prédire du pur bruit à près de 80 % : c'est de la fuite. La seconde donne le résultat honnête, proche du hasard (50 %, avec les fluctuations d'un échantillon de 100 observations).

## C.11 Prévention des erreurs de préparation

**Checklist**

1. ☐ Split **avant** toute transformation qui apprend des données (imputation, scaling, encodage, sélection).
2. ☐ Toute transformation = `fit` sur le train, `transform` sur le test.
3. ☐ Utiliser `Pipeline` + `ColumnTransformer` pour garantir cela automatiquement.
4. ☐ `stratify=y` en classification.
5. ☐ `random_state` fixé pour reproduire.
6. ☐ Aucune feature calculée à partir de la cible ni connue après l'événement.
7. ☐ Déduplication avant le split.
8. ☐ Vérifier la proportion des classes dans chaque jeu.
9. ☐ Garder les données brutes intactes ; documenter chaque transformation.
10. ☐ Se méfier des scores trop beaux : une exactitude de 99 % sur un problème difficile = enquête de fuite.

### Le bon réflexe : `ColumnTransformer` + `Pipeline`

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "revenu", "anciennete", "montant", "dettes"]
nom_cols = ["statut"]
ord_cols = ["niveau"]

prepro = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("nom", OneHotEncoder(handle_unknown="ignore"), nom_cols),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(categories=[["Aucun", "Bac", "Licence", "Master"]]))]), ord_cols),
])

X = credit.drop_duplicates().drop(columns="defaut")
y = credit.drop_duplicates()["defaut"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modele = Pipeline([("prepro", prepro), ("clf", LogisticRegression(max_iter=1000))])
modele.fit(X_train, y_train)                 # toutes les étapes sont ajustées UNIQUEMENT sur le train
from sklearn.metrics import roc_auc_score
print("accuracy test :", round(modele.score(X_test, y_test), 3),
      "| baseline (toujours « pas de défaut ») :", round(1 - y_test.mean(), 3),
      "| AUC :", round(roc_auc_score(y_test, modele.predict_proba(X_test)[:, 1]), 3))
```

Avec ce schéma, `modele.predict(X_nouveau)` applique automatiquement imputation, mise à l'échelle et encodage appris sur le train, sans aucune fuite.

## Exercices C

1. Pourquoi faut-il faire le `train_test_split` avant d'imputer les valeurs manquantes ?
2. Quelle méthode d'encodage pour : (a) `couleur ∈ {rouge, vert, bleu}` ; (b) `taille ∈ {S, M, L, XL}` ; (c) `pays` (150 modalités) ?
3. Quels modèles ont besoin de standardisation : arbre, K-NN, SVM, forêt aléatoire, régression logistique ?
4. Repérez la fuite : « Pour prédire si un patient sera réadmis, on utilise la variable *durée du séjour de réadmission* ».
5. Sur le jeu `credit`, construisez le `Pipeline` de C.11 mais avec `RobustScaler` et `drop="first"` dans l'encodage ; comparez l'AUC (et pas seulement l'accuracy : les défauts sont minoritaires, une baseline qui prédit toujours « pas de défaut » obtient déjà ≈ 0,79).

---
# PARTIE D — Les algorithmes fondamentaux

Pour chaque algorithme : **problème résolu → principe → fonctionnement interne → exemple → avantages/limites → hyperparamètres → quand l'utiliser → code scikit-learn**.

Convention de code pour toute la partie D :

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

Tous les modèles scikit-learn partagent la **même interface** :

```text
modele = Algorithme(hyperparametres)
modele.fit(X_train, y_train)          # apprendre
y_pred = modele.predict(X_test)       # prédire
modele.score(X_test, y_test)          # évaluer (accuracy en classification, R² en régression)
```

---

## D.1 Régression linéaire (*Linear Regression*)

### Problème résolu
**Régression** : prédire une valeur numérique continue (prix, salaire, consommation) à partir de variables numériques, en supposant une relation **linéaire**.

### Principe
Le modèle est une somme pondérée des features :

`ŷ = w₀ + w₁x₁ + w₂x₂ + … + w_p x_p = w₀ + w·x`

- `w₀` : **intercept** (biais), valeur prédite quand toutes les features valent 0.
- `wⱼ` : **coefficient** de la feature `j` : variation de `ŷ` quand `xⱼ` augmente d'une unité, les autres étant fixes.

### Fonctionnement interne
On cherche les `w` qui minimisent la **somme des carrés des résidus** (méthode des moindres carrés, OLS) :

`J(w) = Σᵢ (yᵢ − ŷᵢ)²`

Un **résidu** est `yᵢ − ŷᵢ` (écart entre réalité et prédiction). Élever au carré pénalise fortement les grosses erreurs et rend `J` convexe (un seul minimum).

Deux façons de trouver le minimum :
1. **Équation normale** (solution exacte) : `w = (XᵀX)⁻¹ Xᵀ y`, implémentée via les moindres carrés.
2. **Descente de gradient** (B.8) : itérative, utile pour de très gros volumes.

```python
rng = np.random.default_rng(0)
n = 200
surface = rng.uniform(25, 150, n)
chambres = rng.integers(1, 6, n)
age = rng.uniform(0, 40, n)
prix = 0.8 * surface + 6 * chambres - 0.5 * age + 10 + rng.normal(0, 6, n)      # millions FCFA
maisons = pd.DataFrame({"surface": surface, "chambres": chambres, "age": age, "prix": prix})

# Équation normale « à la main » avec NumPy
X = np.column_stack([np.ones(n), maisons[["surface", "chambres", "age"]].to_numpy()])
w = np.linalg.lstsq(X, maisons["prix"], rcond=None)[0]
print(w.round(2))         # ≈ [10, 0.8, 6, -0.5] : on retrouve les vrais coefficients
```

### Avec scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

Xf = maisons[["surface", "chambres", "age"]]
yf = maisons["prix"]
X_tr, X_te, y_tr, y_te = train_test_split(Xf, yf, test_size=0.25, random_state=0)

reg = LinearRegression().fit(X_tr, y_tr)
print(dict(zip(Xf.columns, reg.coef_.round(2))), round(reg.intercept_, 2))
pred = reg.predict(X_te)
print("MAE :", round(mean_absolute_error(y_te, pred), 2), "| R² :", round(r2_score(y_te, pred), 3))
print(reg.predict(pd.DataFrame({"surface": [90], "chambres": [3], "age": [10]})))     # prix d'une maison type
```

🔍 **Interpréter** : un coefficient de `+0,8` pour `surface` signifie « +1 m² ⇒ +0,8 million, à chambres et âge constants ». Un coefficient négatif sur `age` : plus c'est vieux, moins c'est cher. **Comparer l'importance des variables n'a de sens qu'après standardisation** (sinon les unités faussent la lecture).

### Hypothèses de la régression linéaire

| Hypothèse | Si elle est violée |
|---|---|
| Relation linéaire entre `X` et `y` | prédictions biaisées → ajouter des termes polynomiaux ou changer de modèle |
| Résidus d'espérance nulle, de variance constante (homoscédasticité) | intervalles de confiance faux |
| Résidus indépendants | problème en séries temporelles |
| Pas de forte multicolinéarité | coefficients instables |
| (Pour l'inférence) résidus approximativement normaux | tests de significativité peu fiables |

### Régression polynomiale

On reste linéaire **dans les coefficients** en ajoutant `x², x³…` comme nouvelles features.

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
vraie_f = lambda x: 0.5 * x ** 3 - x
xp_tr = np.linspace(-3, 3, 30).reshape(-1, 1)                # 30 points d'entraînement bruités
yp_tr = vraie_f(xp_tr.ravel()) + rng.normal(0, 3, 30)
xp_te = rng.uniform(-3, 3, (200, 1))                         # 200 points de test, sur la même plage
yp_te = vraie_f(xp_te.ravel()) + rng.normal(0, 3, 200)
for deg in [1, 3, 15]:
    m = make_pipeline(PolynomialFeatures(deg), StandardScaler(), LinearRegression()).fit(xp_tr, yp_tr)
    print(deg, "train R²:", round(m.score(xp_tr, yp_tr), 3), "test R²:", round(m.score(xp_te, yp_te), 3))
# degré 1 : sous-apprentissage (train et test faibles) ; degré 3 : le bon modèle (meilleur test, ≈ 0,59) ;
# degré 15 : le train monte (≈ 0,88) mais le test redescend (≈ 0,50) : surapprentissage
```

### Régularisation : Ridge, Lasso, ElasticNet

Pour contrôler la complexité et l'overfitting, on **pénalise les gros coefficients** :

| Modèle | Coût | Effet |
|---|---|---|
| **Ridge (L2)** | `J + α Σ wⱼ²` | réduit tous les coefficients, stable en cas de colinéarité |
| **Lasso (L1)** | `J + α Σ │wⱼ│` | met certains coefficients **à zéro** : sélection de variables |
| **ElasticNet** | mélange L1 + L2 | compromis |

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
for nom, m in [("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=0.1))]:
    p = make_pipeline(StandardScaler(), m).fit(X_tr, y_tr)
    print(nom, p[-1].coef_.round(2), round(p.score(X_te, y_te), 3))
```

Le paramètre `alpha` règle la force de la pénalité (0 = régression ordinaire). **Toujours standardiser** avant de régulariser.

### Avantages / limites

| ✔ Avantages | ✘ Limites |
|---|---|
| Simple, rapide, très interprétable | n'apprend que des relations linéaires |
| Peu de données requises | sensible aux outliers (carrés) |
| Base de référence (*baseline*) idéale | colinéarité → coefficients instables |
| Extrapole de façon cohérente | n'apprend pas les interactions sans les construire |

### Paramètres importants
`fit_intercept` (par défaut `True`), `alpha` (Ridge/Lasso), `degree` (polynomial).

### 🎯 Quand l'utiliser
Première intention en régression ; besoin d'**expliquer** l'effet de chaque variable ; peu de données ; relation raisonnablement linéaire (prix, consommation, ventes).

---

## D.2 Régression logistique (*Logistic Regression*)

### Problème résolu
**Classification** (surtout binaire) : estimer la **probabilité** qu'une observation appartienne à une classe (défaut de paiement : oui/non).

### Principe
Une combinaison linéaire `z = w₀ + w·x` n'est pas une probabilité (elle peut valoir −50 ou 300). On la passe dans la **fonction sigmoïde** qui l'écrase dans ]0, 1[ :

`σ(z) = 1 / (1 + e⁻ᶻ)`   →   `P(y=1│x) = σ(w₀ + w·x)`

```
σ(z)
1 ┤               ╭──────────
  │            ╭──╯
0.5┤          ╭╯
  │       ╭──╯
0 ┼───────╯
  └───────────────────────── z
        z=0 → 0.5
```

On prédit la classe 1 si `P ≥ 0,5` (seuil modifiable). La frontière de décision `w₀ + w·x = 0` est **linéaire**.

### Fonctionnement interne
On maximise la **vraisemblance** des données, c'est-à-dire qu'on minimise la **perte logistique** (*log loss*, entropie croisée) :

`J = −(1/n) Σ [yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]`

Elle punit lourdement une prédiction confiante et fausse (p=0,99 alors que la vérité est 0). Pas de formule fermée : on utilise une optimisation itérative (descente de gradient, L-BFGS).

**Cote (odds)** et coefficients : `log(p/(1−p)) = w₀ + w·x`. Donc `exp(wⱼ)` est le **facteur multiplicatif de la cote** quand `xⱼ` augmente d'une unité.

```python
# Implémentation minimale avec descente de gradient
def sigmoide(z): return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 2))
y = (X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.5, 300) > 0).astype(int)
w, b, eta = np.zeros(2), 0.0, 0.5
for _ in range(500):
    p = sigmoide(X @ w + b)
    w -= eta * (X.T @ (p - y)) / len(y)           # gradient de la perte logistique
    b -= eta * (p - y).mean()
print(w.round(2), round(b, 2), "accuracy:", ((sigmoide(X @ w + b) > 0.5) == y).mean().round(3))
```

### Avec scikit-learn sur le jeu « credit »

On réutilise le pipeline de la partie C (imputation, standardisation, encodage).

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

rng = np.random.default_rng(42)
n = 2000
revenu = np.round(rng.lognormal(np.log(250_000), 0.45, n), -2)
dettes = np.round(revenu * rng.uniform(0, 2.5, n), -2)
montant = np.round(revenu * rng.uniform(2, 12, n), -3)
anciennete = rng.integers(0, 30, n)
statut = rng.choice(["Salarié", "Indépendant", "Fonctionnaire", "Sans emploi"], n, p=[0.45, 0.25, 0.2, 0.1])
z = (-4.4 + 1.6 * dettes / revenu + 0.12 * montant / revenu - 0.015 * anciennete
     + (statut == "Sans emploi") * 1.2 + (statut == "Indépendant") * 0.3 + rng.normal(0, 0.5, n))
defaut = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
credit = pd.DataFrame({"revenu": revenu, "dettes": dettes, "montant": montant,
                       "anciennete": anciennete, "statut": statut})
credit["ratio_dette"] = credit["dettes"] / credit["revenu"]
credit["ratio_montant"] = credit["montant"] / credit["revenu"]

num = ["revenu", "anciennete", "ratio_dette", "ratio_montant"]
prepro = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), ["statut"])])
logit = Pipeline([("prepro", prepro), ("clf", LogisticRegression(max_iter=1000))])

X_tr, X_te, y_tr, y_te = train_test_split(credit, defaut, test_size=0.25, random_state=0, stratify=defaut)
logit.fit(X_tr, y_tr)
print(classification_report(y_te, logit.predict(X_te), digits=3))

noms = logit[:-1].get_feature_names_out()
coefs = pd.Series(logit[-1].coef_[0], index=noms).sort_values()
print(coefs.round(2))                                   # coefficients (échelle log-odds)
print(np.exp(coefs).round(2))                           # facteurs multiplicatifs de la cote

proba = logit.predict_proba(X_te)[:, 1]                 # probabilité de défaut
print(proba[:5].round(3))
```

🔍 **Interpréter** : `exp(coef) = 2,0` pour `ratio_dette` signifie « +1 écart-type de ratio de dette double la cote de défaut, toutes choses égales par ailleurs ». `predict_proba` donne la probabilité ; `predict` applique le seuil 0,5. On peut **changer le seuil** pour arbitrer précision et rappel (partie E).

### Multi-classes
Scikit-learn gère le cas à 3+ classes automatiquement (stratégie multinomiale « softmax »).

### Régularisation
Paramètre **`C`** = inverse de la force de régularisation (`C` petit ⇒ forte régularisation). `penalty="l2"` par défaut.

### Avantages / limites

| ✔ Avantages | ✘ Limites |
|---|---|
| Probabilités calibrées, interprétable (odds) | frontière linéaire uniquement (sauf features polynomiales) |
| Rapide, robuste, bon *baseline* | sensible à la colinéarité |
| Régularisable | nécessite une mise à l'échelle |
| Peu d'hyperparamètres | moins performant que les ensembles sur données complexes |

### Paramètres importants
`C`, `penalty` (`l1`/`l2`), `class_weight="balanced"` (classes déséquilibrées), `solver`, `max_iter`.

### 🎯 Quand l'utiliser
Classification binaire de référence ; besoin de **probabilités** et d'explications (crédit, médecine, marketing) ; données tabulaires de taille modérée.

---

## D.3 K plus proches voisins (*K-Nearest Neighbors, K-NN*)

### Problème résolu
Classification ou régression **sans hypothèse** sur la forme de la relation.

### Principe
« Dis-moi qui sont tes voisins, je te dirai qui tu es. » Pour prédire une nouvelle observation :
1. calculer sa **distance** à toutes les observations d'entraînement ;
2. retenir les **K plus proches** ;
3. **classification** : vote majoritaire ; **régression** : moyenne de leurs valeurs.

### Fonctionnement interne
- **Aucun entraînement** : l'algorithme mémorise simplement les données (*lazy learner*). Tout le calcul est fait à la prédiction.
- **Distances** : euclidienne `√Σ(aᵢ−bᵢ)²` (défaut), Manhattan `Σ│aᵢ−bᵢ│`, Minkowski (généralisation), cosinus.
- **Pondération** : `weights="distance"` donne plus d'importance aux voisins proches.

```python
# K-NN « à la main » pour comprendre
def knn_predire(X_train, y_train, x, k=3):
    distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
    voisins = np.argsort(distances)[:k]
    valeurs, comptes = np.unique(y_train[voisins], return_counts=True)
    return valeurs[comptes.argmax()]

Xt = np.array([[1, 1], [1, 2], [2, 1], [6, 6], [7, 6], [6, 7]])
yt = np.array(["A", "A", "A", "B", "B", "B"])
print(knn_predire(Xt, yt, np.array([2, 2])), knn_predire(Xt, yt, np.array([5, 5])))   # A B
```

### Effet de K

| K | Frontière | Risque |
|---|---|---|
| petit (1) | très irrégulière | **overfitting** (sensible au bruit) |
| grand | très lisse | **underfitting** (tout devient la classe majoritaire) |

### Avec scikit-learn : classification des iris

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

for k in [1, 3, 5, 15, 50]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_tr, y_tr)
    print(f"k={k:<3} train={m.score(X_tr, y_tr):.3f}  test={m.score(X_te, y_te):.3f}")

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_tr, y_tr)
print(iris.target_names[knn.predict([[5.1, 3.5, 1.4, 0.2]])])        # ['setosa']
```

Choisir `K` : validation croisée (partie E). Valeur impaire en binaire pour éviter les égalités ; ordre de grandeur `√n`.

⚠️ **Pièges**
- **Standardisation indispensable** : sinon la variable à grande échelle domine la distance.
- **Malédiction de la dimension** : en haute dimension, toutes les distances deviennent presque égales ; K-NN se dégrade (réduire par ACP/sélection).
- **Lent à la prédiction** et gourmand en mémoire sur gros jeux.

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Très simple, aucune hypothèse | prédiction lente (`O(n)` par exemple) |
| Frontières non linéaires | sensible à l'échelle et aux variables inutiles |
| S'adapte à la classification et à la régression | souffre en grande dimension et si classes déséquilibrées |

### Paramètres importants
`n_neighbors`, `weights` (`uniform`/`distance`), `metric`/`p`, `algorithm` (`auto`, `kd_tree`, `ball_tree`).

### 🎯 Quand l'utiliser
Petit jeu de données, peu de variables bien mises à l'échelle, recommandations par similarité, baseline rapide à mettre en œuvre.

---

## D.4 Arbres de décision (*Decision Trees*)

### Problème résolu
Classification et régression, avec un modèle **lisible comme une suite de questions**.

### Principe
Un arbre pose une série de questions « oui/non » sur les features (`revenu < 180 000 ?`) pour découper l'espace en régions homogènes.

```
                 [ratio_dette < 0.9 ?]
                 /                  \
              oui                    non
     [statut = Sans emploi ?]      [anciennete < 5 ?]
        /        \                  /        \
   défaut=22 %  défaut=8 %    défaut=45 %   défaut=29 %
```

- **Nœud interne** : une question ; **branche** : réponse ; **feuille** : prédiction (classe majoritaire, ou moyenne en régression).

### Fonctionnement interne : comment choisit-on la meilleure question ?
À chaque nœud, l'algorithme (CART) teste toutes les features et tous les seuils possibles, et retient la coupure qui **réduit le plus l'impureté**.

**Impureté de Gini** d'un nœud : `G = 1 − Σₖ pₖ²` (`pₖ` = proportion de la classe `k`).
- Nœud pur (100 % classe A) : `G = 0`.
- Nœud 50/50 : `G = 0,5` (maximum en binaire).

**Entropie** : `H = −Σₖ pₖ log₂ pₖ` (alternative, `criterion="entropy"`).

**Gain** d'une coupure = impureté du parent − moyenne pondérée des impuretés des enfants.

```python
def gini(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return 1 - (p ** 2).sum()

parent = np.array([0] * 50 + [1] * 50)
gauche = np.array([0] * 45 + [1] * 5)
droite = np.array([0] * 5 + [1] * 45)
gain = gini(parent) - (len(gauche) * gini(gauche) + len(droite) * gini(droite)) / len(parent)
print(round(gini(parent), 3), round(gini(gauche), 3), round(gain, 3))     # 0.5 0.18 0.32
```

Pour la **régression**, le critère est la réduction de la variance (erreur quadratique) ; la feuille prédit la moyenne.

Le processus est **récursif** et **glouton** : on coupe, puis on recommence dans chaque enfant, jusqu'à un critère d'arrêt (profondeur max, effectif minimal, pureté).

### Avec scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

for prof in [1, 2, 3, 5, 10, None]:
    arbre = DecisionTreeClassifier(max_depth=prof, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={str(prof):<5} train={arbre.score(X_tr, y_tr):.3f}  test={arbre.score(X_te, y_te):.3f}")
# Un arbre sans limite atteint train=1.0 : il mémorise (overfitting)

arbre = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
print(export_text(arbre, feature_names=list(X.columns)))               # règles lisibles
imp = pd.Series(arbre.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(5).round(3))
```

### Contrôle de l'overfitting (élagage)
Un arbre non contraint crée une feuille par observation. On le limite avec : `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `ccp_alpha` (élagage par coût-complexité).

### Avantages / limites

| ✔ | ✘ |
|---|---|
| **Très interprétable** (règles si-alors) | **instable** : une petite modification des données change l'arbre |
| Aucune mise à l'échelle nécessaire | **overfitting** facile |
| Gère numérique + catégoriel, relations non linéaires, interactions | frontières « en escaliers » (parallèles aux axes) |
| Robuste aux outliers en entrée | extrapole mal (régression : constant hors plage) |

### Paramètres importants
`max_depth`, `min_samples_leaf`, `criterion` (`gini`/`entropy`/`log_loss`), `class_weight`, `ccp_alpha`.

### 🎯 Quand l'utiliser
Besoin d'expliquer les décisions à un non-technicien ; exploration des interactions ; brique de base des forêts et du boosting. Rarement le modèle final seul.

---

## D.5 Forêt aléatoire (*Random Forest*)

### Problème résolu
Dépasser l'instabilité et le surapprentissage d'un arbre unique, en conservant sa souplesse.

### Principe : la sagesse des foules
Entraîner **de nombreux arbres différents** puis **voter** (classification) ou **moyenner** (régression). Des erreurs peu corrélées se compensent.

### Fonctionnement interne
Deux sources d'aléa rendent les arbres différents :
1. **Bagging** (*bootstrap aggregating*) : chaque arbre est entraîné sur un échantillon tiré **avec remise** de la taille du jeu (≈ 63 % d'observations distinctes ; les ≈ 37 % restantes sont les *out-of-bag*).
2. **Sous-échantillonnage des features** : à chaque nœud, seul un sous-ensemble aléatoire de variables (`max_features`, ≈ √p en classification) est candidat.

Résultat : variance réduite, biais quasi identique à celui d'un arbre profond.

```python
from sklearn.ensemble import RandomForestClassifier

foret = RandomForestClassifier(n_estimators=300, max_depth=None, oob_score=True,
                               n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print("test :", round(foret.score(X_te, y_te), 3))
print("score out-of-bag (estimation gratuite de la généralisation) :", round(foret.oob_score_, 3))

imp = pd.Series(foret.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(6).round(3))
```

**Importance des variables** : baisse moyenne d'impureté apportée par chaque variable. ⚠️ Biaisée en faveur des variables continues/à nombreuses modalités ; préférez l'**importance par permutation** :

```python
from sklearn.inspection import permutation_importance
res = permutation_importance(foret, X_te, y_te, n_repeats=10, random_state=0)
print(pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False).head(5).round(3))
```

Comparaison arbre / forêt :

```python
for n_arbres in [1, 10, 100, 300]:
    f = RandomForestClassifier(n_estimators=n_arbres, random_state=0).fit(X_tr, y_tr)
    print(n_arbres, round(f.score(X_te, y_te), 3))
```

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Très performante « out of the box » | moins interprétable qu'un arbre |
| Robuste au surapprentissage (plus d'arbres ≠ plus d'overfit) | plus lente et plus lourde en mémoire |
| Pas de mise à l'échelle, tolère les NA (selon versions) et outliers | n'extrapole pas hors de la plage d'entraînement |
| Score OOB, importances | surpassée en précision par le boosting bien réglé |

### Paramètres importants

| Paramètre | Effet |
|---|---|
| `n_estimators` | nombre d'arbres (100–500 ; plus = plus stable, plus lent) |
| `max_depth` | profondeur maximale |
| `max_features` | variables candidates par nœud (`sqrt` par défaut) |
| `min_samples_leaf` | lisse (augmenter contre l'overfit) |
| `class_weight` | classes déséquilibrées |
| `n_jobs=-1` | parallélisation |

### 🎯 Quand l'utiliser
Excellent **premier modèle puissant** sur données tabulaires, avec peu de réglages ; importance des variables ; données mélangeant types et échelles.

---

## D.6 Machines à vecteurs de support (*SVM*)

### Problème résolu
Classification (et régression, `SVR`) avec une frontière de **marge maximale**, y compris non linéaire.

### Principe
Parmi toutes les droites (hyperplans) qui séparent deux classes, la SVM choisit celle qui laisse la **plus grande marge**, c'est-à-dire la plus grande distance aux points les plus proches de chaque classe.

```
 classe A   ○ ○                       ┊ marge ┊
          ○   ○     ○      ━━━━━━━━━━━━━━━━━━  ← frontière
                         ●   ●
                          ●        ● classe B
 Les points sur le bord de la marge sont les « vecteurs de support » :
 ce sont les SEULS qui déterminent la frontière.
```

### Fonctionnement interne
- **Marge** : la frontière `w·x + b = 0`, marges `w·x + b = ±1`, largeur `2/‖w‖` ; maximiser la marge ⇒ minimiser `‖w‖²`.
- **Marge souple** (*soft margin*) : on tolère quelques erreurs. Le paramètre **`C`** arbitre : `C` grand = peu d'erreurs tolérées (risque d'overfit), `C` petit = marge large et plus de tolérance.
- **Astuce du noyau** (*kernel trick*) : pour des données non séparables linéairement, on les projette implicitement dans un espace de dimension supérieure où elles le deviennent, sans calculer explicitement la projection.

| Noyau | Frontière | Usage |
|---|---|---|
| `linear` | droite/hyperplan | grandes dimensions, texte |
| `rbf` (gaussien, défaut) | courbe flexible | cas général |
| `poly` | polynomiale | interactions de degré fixe |

Le noyau RBF : `K(x, x') = exp(−γ‖x − x'‖²)`. Le paramètre **`gamma`** contrôle la « portée » : grand ⇒ frontière très locale (overfit) ; petit ⇒ frontière lisse.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=0)       # 2 demi-lunes entremêlées
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)

for noyau in ["linear", "rbf"]:
    s = make_pipeline(StandardScaler(), SVC(kernel=noyau)).fit(Xm_tr, ym_tr)
    print(noyau, round(s.score(Xm_te, ym_te), 3))          # le noyau RBF gère la non-linéarité

for C, gamma in [(0.1, 0.1), (1, 1), (100, 100)]:
    s = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(Xm_tr, ym_tr)
    print(f"C={C:<5} gamma={gamma:<5} train={s.score(Xm_tr, ym_tr):.3f} test={s.score(Xm_te, ym_te):.3f}")
# C=100, gamma=100 : train proche de 1, test plus bas → surapprentissage
```

Pour obtenir des probabilités : `SVC(probability=True)` (plus lent, calibrage par validation croisée interne).

⚠️ **Pièges** : standardisation **obligatoire** ; temps d'entraînement `O(n²)`–`O(n³)` : impraticable au-delà de ~100 000 observations (utiliser `LinearSVC` ou `SGDClassifier`) ; peu interprétable.

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Efficace en grande dimension | lent sur gros jeux de données |
| Frontières non linéaires via noyaux | choix `C`/`gamma` délicat |
| Robuste (seuls les vecteurs de support comptent) | pas de probabilités natives |
| Fonctionne avec peu de données | sensible à l'échelle |

### Paramètres importants
`C`, `kernel`, `gamma`, `degree` (poly), `class_weight`.

### 🎯 Quand l'utiliser
Jeux de taille petite à moyenne, dimension élevée (texte, bio-informatique), frontière complexe, classes bien séparées.

---

## D.7 Naive Bayes

### Problème résolu
Classification rapide, particulièrement bien adaptée au **texte** (spam, sentiment).

### Principe
Application du **théorème de Bayes** (B.3) :

`P(classe │ x₁,…,x_p) ∝ P(classe) · Πⱼ P(xⱼ │ classe)`

L'hypothèse « naïve » : les features sont **indépendantes conditionnellement à la classe**. Fausse en pratique, mais l'algorithme marche souvent remarquablement bien malgré elle.

### Fonctionnement interne
1. Apprentissage : estimer les **probabilités a priori** `P(classe)` (fréquences) et les **vraisemblances** `P(xⱼ│classe)` pour chaque feature.
2. Prédiction : calculer pour chaque classe le produit (en pratique la somme des logarithmes, pour éviter les sous-dépassements numériques) et choisir la classe maximale.

| Variante | Features | Vraisemblance |
|---|---|---|
| `GaussianNB` | continues | loi normale par classe |
| `MultinomialNB` | comptages (mots) | multinomiale |
| `BernoulliNB` | binaires (mot présent ou pas) | Bernoulli |

### Exemple concret : détection de spam

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "gagnez de l'argent facilement cliquez ici", "offre gratuite limitée gagnez un prix",
    "félicitations vous avez gagné un voyage gratuit", "cliquez pour réclamer votre prix gratuit",
    "argent rapide sans effort offre exclusive", "promo gratuite cliquez maintenant",
    "réunion demain à 10h avec l'équipe", "peux-tu m'envoyer le rapport de ventes",
    "voici le compte rendu de la réunion", "rendez-vous pour le déjeuner jeudi",
    "merci pour ton aide sur le projet", "le planning de la semaine prochaine est joint",
]
etiquettes = ["spam"] * 6 + ["ham"] * 6

spam = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(messages, etiquettes)
tests = ["offre gratuite cliquez ici", "peux-tu envoyer le planning de la réunion"]
print(spam.predict(tests))                      # ['spam' 'ham']
print(spam.predict_proba(tests).round(3))
```

Le paramètre `alpha` est le **lissage de Laplace** : il évite qu'un mot jamais vu dans une classe donne une probabilité nulle et annule tout le produit.

Exemple sur données continues :

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_tr, y_tr)                  # jeu cancer de D.4
print("GaussianNB :", round(gnb.score(X_te, y_te), 3))
```

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Extrêmement rapide, peu de données requises | hypothèse d'indépendance irréaliste |
| Bon baseline pour le texte | probabilités mal calibrées (trop extrêmes) |
| Gère de nombreuses features | ne capture pas les interactions |
| Peu d'hyperparamètres | `GaussianNB` suppose des distributions gaussiennes |

### Paramètres importants
`alpha` (lissage), `var_smoothing` (Gaussien), `priors`.

### 🎯 Quand l'utiliser
Classification de texte, baseline instantané, très grand volume de données, flux en ligne (`partial_fit`).

---

## D.8 K-Means (clustering)

### Problème résolu
**Non supervisé** : regrouper des observations en `K` **clusters** (groupes) homogènes sans connaître d'étiquettes. Exemple : segmenter des clients.

### Principe
Chaque cluster est représenté par son **centroïde** (centre de gravité). Chaque point appartient au cluster dont le centroïde est le plus proche.

### Fonctionnement interne (algorithme de Lloyd)
1. Choisir `K` centroïdes initiaux (`k-means++` : initialisation intelligente, bien dispersée).
2. **Affectation** : chaque point rejoint le centroïde le plus proche.
3. **Mise à jour** : chaque centroïde devient la moyenne des points de son cluster.
4. Répéter 2–3 jusqu'à stabilisation.

L'objectif minimisé est l'**inertie** (somme des carrés des distances aux centroïdes) : `Σ ‖xᵢ − μ_{c(i)}‖²`.

```python
def kmeans(X, k, n_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)     # broadcasting (n, k)
        etiq = d.argmin(axis=1)
        centres = np.array([X[etiq == j].mean(axis=0) if (etiq == j).any() else centres[j]
                            for j in range(k)])
    return etiq, centres
```

### Avec scikit-learn : segmentation de clients

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(3)
n = 600
clients = pd.DataFrame({
    "depense_annuelle": np.r_[rng.normal(150, 30, 250), rng.normal(600, 90, 200), rng.normal(1500, 200, 150)],
    "visites_par_mois": np.r_[rng.normal(2, 1, 250), rng.normal(8, 2, 200), rng.normal(20, 3, 150)],
}).clip(lower=0)

Xc = StandardScaler().fit_transform(clients)                 # standardiser est essentiel
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xc)
clients["segment"] = km.labels_
print(clients.groupby("segment").agg(["mean", "count"]).round(1))      # profil de chaque segment
```

### Choisir K

**Méthode du coude** (*elbow*) : tracer l'inertie en fonction de `K` ; chercher le « coude » où le gain devient marginal.
**Score de silhouette** : pour chaque point, `s = (b − a) / max(a, b)`, `a` = distance moyenne à son cluster, `b` = distance moyenne au cluster voisin le plus proche. Entre −1 et 1 ; proche de 1 = bien classé.

```python
for k in range(2, 7):
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xc)
    print(k, "inertie:", round(m.inertia_, 1), "silhouette:", round(silhouette_score(Xc, m.labels_), 3))
# La silhouette de K=2 et K=3 est quasi identique (les deux plus gros groupes sont assez proches) ;
# l'inertie, elle, chute brutalement de 2 à 3 (213 → 64) puis ne gagne presque plus : le « coude » est à K=3
```

⚠️ **Pièges**
- Il faut **fixer K** à l'avance (pas de « bonne » valeur automatique).
- Suppose des clusters **sphériques et de tailles comparables** ; échoue sur des formes allongées ou en anneau (essayez DBSCAN).
- Sensible à l'initialisation (d'où `n_init=10`) et aux outliers.
- Sensible aux échelles : standardiser.
- Les clusters ne sont pas des « vérités » : il faut les **interpréter** (profil moyen de chaque groupe).

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Simple, rapide, passe à l'échelle | K à fixer, clusters convexes |
| Résultats faciles à expliquer (centroïdes) | sensible aux outliers/échelle/initialisation |

### Paramètres importants
`n_clusters`, `init="k-means++"`, `n_init`, `max_iter`, `random_state`.

### 🎯 Quand l'utiliser
Segmentation de clientèle, compression d'images (quantification de couleurs), pré-regroupement, exploration de structure.

---

## D.9 Clustering hiérarchique

### Problème résolu
Regrouper des observations en construisant un **arbre de regroupements** (dendrogramme), sans fixer K au départ.

### Principe et fonctionnement
**Approche agglomérative** (ascendante) :
1. Chaque point est son propre cluster.
2. Fusionner les deux clusters les plus proches.
3. Recommencer jusqu'à un seul cluster.

Le **dendrogramme** retrace les fusions ; la hauteur d'une fusion = distance entre les clusters fusionnés. En **coupant** à une hauteur donnée, on obtient K clusters.

**Critère de liaison (*linkage*)** : comment mesurer la distance entre deux clusters ?

| Linkage | Distance entre clusters | Tendance |
|---|---|---|
| `single` | entre les deux points les plus proches | chaînes allongées |
| `complete` | entre les deux points les plus éloignés | clusters compacts |
| `average` | moyenne de toutes les paires | intermédiaire |
| **`ward`** | minimise l'augmentation de la variance interne | clusters équilibrés (le plus utilisé, distance euclidienne) |

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

Xs, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=1)

agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(Xs)
print(np.bincount(agg.labels_))                      # taille des 3 clusters

Z = linkage(Xs, method="ward")                       # matrice des fusions
plt.figure(figsize=(8, 3.5)); dendrogram(Z, no_labels=True); plt.title("Dendrogramme (Ward)"); plt.show()
labels = fcluster(Z, t=3, criterion="maxclust")      # couper pour obtenir 3 clusters
```

Lecture : une **grande hauteur** entre deux fusions successives indique une séparation naturelle ; couper juste en dessous du plus grand saut.

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Pas de K préalable, vue hiérarchique | coût `O(n²)` mémoire : limité à quelques milliers de points |
| Dendrogramme interprétable | décisions de fusion irréversibles |
| Peut capturer des formes variées (single) | sensible au bruit et au choix du linkage |

### Paramètres importants
`n_clusters` ou `distance_threshold`, `linkage`, `metric`.

### 🎯 Quand l'utiliser
Petit à moyen jeu de données ; besoin d'une taxonomie (espèces, gènes, documents) ; choisir K en regardant le dendrogramme.

---

## D.10 Analyse en composantes principales (ACP / *PCA*)

### Problème résolu
**Réduire la dimension** d'un jeu (passer de 50 variables à 3 composantes) en **conservant le maximum d'information** (variance). Sert à visualiser, compresser, débruiter, accélérer.

### Principe
Trouver de **nouveaux axes** (composantes principales), combinaisons linéaires des variables d'origine, qui sont :
1. **orthogonaux** (non corrélés) ;
2. ordonnés par **variance expliquée** décroissante : le 1er axe capture le plus de variabilité, le 2e le plus de ce qui reste, etc.

Image : photographier un nuage de points allongé sous l'angle qui montre sa plus grande étendue.

### Fonctionnement interne
1. **Centrer** (et en général **standardiser**) les données.
2. Calculer la **matrice de covariance** `C` des features.
3. Calculer ses **valeurs propres** `λₖ` et **vecteurs propres** `vₖ` (B.1.2) : `vₖ` = direction de la composante `k`, `λₖ` = variance portée par cette direction.
4. Garder les `d` plus grandes valeurs propres et **projeter** : `Z = X_centré @ [v₁ … v_d]`.

Part de variance expliquée par la composante `k` : `λₖ / Σ λ`.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

vin = load_wine(as_frame=True)
Xv = StandardScaler().fit_transform(vin.data)               # 13 variables chimiques

# PCA « à la main »
C = np.cov(Xv, rowvar=False)
vals, vecs = np.linalg.eigh(C)
ordre = np.argsort(vals)[::-1]; vals, vecs = vals[ordre], vecs[:, ordre]
print((vals / vals.sum()).round(3)[:4])                     # part de variance des 4 premières composantes

# avec scikit-learn
pca = PCA().fit(Xv)
print(pca.explained_variance_ratio_.round(3)[:4])           # mêmes valeurs
print(np.cumsum(pca.explained_variance_ratio_).round(3)[:5])    # variance cumulée

p2 = PCA(n_components=2)
Z = p2.fit_transform(Xv)                                    # 178 × 2
print(Z.shape)
```

**Combien de composantes garder ?** Lire la **variance cumulée** : par exemple 90–95 % ; ou `PCA(n_components=0.95)` choisit automatiquement. Le graphique d'éboulis (*scree plot*) aide.

**Interpréter les composantes** : les **chargements** (`pca.components_`) indiquent le poids de chaque variable d'origine dans la composante.

```python
charge = pd.DataFrame(p2.components_.T, index=vin.data.columns, columns=["PC1", "PC2"])
print(charge.round(2).sort_values("PC1").iloc[[0, 1, -2, -1]])
```

Usage typique : visualiser les classes en 2D.

```python
plt.figure(figsize=(5, 4))
plt.scatter(Z[:, 0], Z[:, 1], c=vin.target, cmap="viridis", s=18)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Vins projetés sur 2 composantes"); plt.show()
```

⚠️ **Pièges**
- **Standardiser avant** (sinon la variable de plus grande échelle domine).
- L'ACP est **linéaire** et **non supervisée** : elle maximise la variance, pas la séparation des classes.
- Les composantes sont plus difficiles à expliquer que les variables d'origine.
- Appliquer `fit` sur le train uniquement, puis `transform` sur le test.

### Avantages / limites

| ✔ | ✘ |
|---|---|
| Réduit le bruit, la colinéarité, le temps de calcul | relations non linéaires perdues (voir t-SNE, UMAP) |
| Visualisation 2D/3D | composantes peu interprétables |
| Décorrèle les variables | sensible à l'échelle et aux outliers |

### Paramètres importants
`n_components` (entier ou part de variance), `whiten`, `svd_solver`.

### 🎯 Quand l'utiliser
Visualiser des données de grande dimension, éliminer la multicolinéarité avant un modèle linéaire, compresser (images), préparer K-Means / K-NN en grande dimension.

---

## D.11 Tableau comparatif des algorithmes

| Algorithme | Type | Non-linéaire | Échelle requise | Interprétable | Vitesse | Gros volumes | Remarque |
|---|---|---|---|---|---|---|---|
| Régression linéaire | Régr. | non | oui (si régularisé) | ★★★ | ★★★ | ✔ | baseline, explicabilité |
| Régression logistique | Classif. | non | oui | ★★★ | ★★★ | ✔ | probabilités |
| K-NN | Les deux | oui | **oui** | ★★ | ★ (prédiction) | ✘ | simple, sensible aux échelles |
| Arbre de décision | Les deux | oui | non | ★★★ | ★★★ | ✔ | surapprend |
| Random Forest | Les deux | oui | non | ★ | ★★ | ✔ | très bon par défaut |
| SVM | Les deux | oui (noyau) | **oui** | ★ | ★ | ✘ | petite/moyenne taille |
| Naive Bayes | Classif. | – | non | ★★ | ★★★ | ✔ | texte |
| K-Means | Clustering | – | **oui** | ★★ | ★★★ | ✔ | K à fixer |
| Hiérarchique | Clustering | – | oui | ★★ | ★ | ✘ | dendrogramme |
| ACP | Réduction | non | **oui** | ★★ | ★★★ | ✔ | pré-traitement/visualisation |

### Comment choisir ?

```
Cible connue ? ── oui ──▶ supervisé
   │                        ├─ cible numérique ──▶ régression : linéaire (baseline) → forêt / boosting
   │                        └─ cible catégorielle ──▶ classification : logistique (baseline) → forêt / boosting
   │                              (texte ? → Naive Bayes / logistique sur TF-IDF)
   └─ non ──▶ non supervisé
                ├─ regrouper ──▶ K-Means (K connu) / hiérarchique (K inconnu, petit jeu)
                └─ simplifier / visualiser ──▶ ACP
```

**Stratégie professionnelle** : toujours commencer par une **baseline** simple (régression linéaire/logistique, ou modèle naïf), puis tester des modèles plus puissants et ne retenir de la complexité que si elle améliore réellement la performance **mesurée proprement**.

## Exercices D

1. Entraînez une régression linéaire sur `load_diabetes`. Quels sont les 2 coefficients les plus forts après standardisation ?
2. Sur `load_breast_cancer`, comparez régression logistique, K-NN (k=5), arbre (profondeur 4), forêt, SVM et Naive Bayes avec la même séparation train/test. Classez-les. Lesquels nécessitent une standardisation ?
3. Faites varier `K` de 1 à 30 pour K-NN sur les iris et tracez les scores train/test. Où est l'overfitting ? l'underfitting ?
4. Avec `make_moons`, comparez la régression logistique, un arbre de profondeur 5 et un SVM RBF. Pourquoi la régression logistique échoue-t-elle ?
5. Segmentez `clients` (D.8) avec K-Means pour K=2…6 ; justifiez votre choix de K par la silhouette.
6. Appliquez l'ACP sur `load_digits` (64 pixels). Combien de composantes pour conserver 90 % de la variance ?
7. Écrivez votre propre fonction `gini` et calculez le gain d'une coupure sur un petit jeu.

---
# PARTIE E — Évaluation des modèles

> Un modèle sans évaluation rigoureuse est une opinion. Cette partie apprend à **mesurer**, mais surtout à **interpréter** et à **éviter de se mentir**.

## E.0 Principe : évaluer sur des données non vues

La seule performance qui compte est celle sur des **données que le modèle n'a jamais vues**. L'erreur sur le train est optimiste par construction.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

## E.1 Métriques de classification

### E.1.1 Matrice de confusion (*Confusion Matrix*)

Pour un problème binaire (classe positive = ce qu'on cherche, par exemple « fraude ») :

|  | **Prédit négatif** | **Prédit positif** |
|---|---|---|
| **Réel négatif** | **VN** vrais négatifs | **FP** faux positifs (fausse alerte, erreur de type I) |
| **Réel positif** | **FN** faux négatifs (cas manqué, erreur de type II) | **VP** vrais positifs |

Dans scikit-learn : `confusion_matrix(y_vrai, y_pred)` renvoie `[[VN, FP], [FN, VP]]` (lignes = réel, colonnes = prédit).

### E.1.2 Les métriques

| Métrique | Formule | Question posée |
|---|---|---|
| **Accuracy** (exactitude) | `(VP+VN) / total` | quelle part de prédictions est correcte ? |
| **Precision** (précision) | `VP / (VP+FP)` | quand j'annonce « positif », ai-je raison ? |
| **Recall** (rappel, sensibilité) | `VP / (VP+FN)` | quelle part des vrais positifs ai-je trouvée ? |
| **Spécificité** | `VN / (VN+FP)` | quelle part des vrais négatifs ai-je bien écartée ? |
| **F1-score** | `2·P·R / (P+R)` | compromis (moyenne harmonique) précision/rappel |

La moyenne harmonique du F1 est tirée vers la plus petite des deux valeurs : impossible d'avoir un bon F1 avec un rappel nul.

### Calcul sur un exemple

Détection de fraude : sur 1 000 transactions, 20 sont frauduleuses.

```python
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

y_vrai = np.array([1] * 20 + [0] * 980)
y_pred = np.array([1] * 12 + [0] * 8 + [1] * 10 + [0] * 970)      # 12 VP, 8 FN, 10 FP, 970 VN
print(confusion_matrix(y_vrai, y_pred))
# [[970  10]
#  [  8  12]]
print("accuracy :", accuracy_score(y_vrai, y_pred))                 # 0.982
print("precision:", round(precision_score(y_vrai, y_pred), 3))      # 12/22 = 0.545
print("recall   :", round(recall_score(y_vrai, y_pred), 3))         # 12/20 = 0.6
print("f1       :", round(f1_score(y_vrai, y_pred), 3))             # 0.571
print(classification_report(y_vrai, y_pred, digits=3))
```

### 🔍 Interpréter : le piège de l'accuracy

Ici l'accuracy est de **98,2 %** : impressionnant ! Pourtant, un modèle qui répond **toujours « pas de fraude »** obtiendrait **98 %**, et ne détecterait **aucune** fraude.

```python
print(accuracy_score(y_vrai, np.zeros(1000)))        # 0.98 sans rien détecter
print(recall_score(y_vrai, np.zeros(1000)))          # 0.0
```

> ⚠️ **Sur des classes déséquilibrées, l'accuracy ne vaut presque rien.** Regardez précision, rappel, F1 et la matrice de confusion.

### Quelle métrique choisir ? (selon le coût des erreurs)

| Situation | Erreur la plus grave | Métrique à privilégier |
|---|---|---|
| Dépistage d'une maladie grave | FN (malade non détecté) | **Rappel** |
| Filtre anti-spam | FP (courriel important mis en spam) | **Précision** |
| Détection de fraude | FN coûteux, mais trop de FP épuisent les équipes | **F1** / rappel à précision donnée |
| Classes équilibrées, coûts égaux | — | Accuracy |
| Recommandation « top K » | pertinence des K premiers | précision@K |

### Seuil de décision, courbes précision-rappel et ROC

La classification repose sur une **probabilité** et un **seuil** (0,5 par défaut). Faire varier le seuil déplace l'équilibre précision/rappel.

| Seuil | Effet |
|---|---|
| **bas** (0,2) | on prédit « positif » plus souvent : ↑ rappel, ↓ précision |
| **haut** (0,8) | on est exigeant : ↑ précision, ↓ rappel |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
proba = m.predict_proba(X_te)[:, 1]

for seuil in [0.2, 0.5, 0.8, 0.95]:
    p = (proba >= seuil).astype(int)
    print(f"seuil={seuil:<4} precision={precision_score(y_te, p):.3f} recall={recall_score(y_te, p):.3f}")
print("AUC-ROC :", round(roc_auc_score(y_te, proba), 3))
```

**AUC-ROC** (aire sous la courbe ROC) : probabilité qu'un positif tiré au hasard reçoive un score supérieur à celui d'un négatif tiré au hasard. 0,5 = hasard ; 1 = parfait. Avec classes très déséquilibrées, préférez la **précision-rappel** (`average_precision_score`).

### Classification multi-classes

`classification_report` donne les métriques par classe. Moyennes : **macro** (non pondérée, traite les classes à égalité), **weighted** (pondérée par l'effectif), **micro** (globale).

## E.2 Métriques de régression

Soit `eᵢ = yᵢ − ŷᵢ` les erreurs.

| Métrique | Formule | Unité | Propriété |
|---|---|---|---|
| **MAE** (erreur absolue moyenne) | `(1/n) Σ │eᵢ│` | celle de `y` | intuitive, robuste aux outliers |
| **MSE** (erreur quadratique moyenne) | `(1/n) Σ eᵢ²` | `y²` | punit fortement les grosses erreurs, dérivable |
| **RMSE** | `√MSE` | celle de `y` | comme l'écart-type des erreurs |
| **R²** | `1 − Σeᵢ² / Σ(yᵢ−ȳ)²` | sans unité | part de variance expliquée |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

y_vrai = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 210, 230, 320])
print("MAE :", mean_absolute_error(y_vrai, y_pred))             # 14.0
print("MSE :", mean_squared_error(y_vrai, y_pred))              # 220.0
print("RMSE:", round(root_mean_squared_error(y_vrai, y_pred), 2))   # 14.83
print("R²  :", round(r2_score(y_vrai, y_pred), 4))              # 0.956
```

### 🔍 Interpréter

- **MAE = 14** (en millions de FCFA par exemple) : « en moyenne je me trompe de 14 ». C'est la métrique la plus parlante à présenter à un décideur.
- **RMSE ≥ MAE** toujours. Un grand écart RMSE/MAE (ici 14,8 vs 14) révèle quelques **grosses erreurs**.
- **R²** : 0 = pas mieux que prédire la moyenne ; 1 = parfait ; **peut être négatif** sur le test si le modèle est pire que la moyenne. R² = 0,956 : le modèle explique 95,6 % de la variance.
- Toujours **comparer à une baseline** : un `DummyRegressor` qui prédit la moyenne.

```python
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25, random_state=0)
base = DummyRegressor(strategy="mean").fit(Xd_tr, yd_tr)
lin = LinearRegression().fit(Xd_tr, yd_tr)
print("MAE baseline :", round(mean_absolute_error(yd_te, base.predict(Xd_te)), 1))
print("MAE linéaire :", round(mean_absolute_error(yd_te, lin.predict(Xd_te)), 1))
print("R² linéaire  :", round(lin.score(Xd_te, yd_te), 3))
```

⚠️ **Pièges**
- Le R² augmente mécaniquement quand on ajoute des variables : utilisez le R² ajusté ou la validation croisée.
- Un bon R² ne prouve pas que le modèle est bon (vérifiez les résidus : doivent être sans structure).
- MSE/RMSE sont dominées par les outliers ; MAE est plus robuste.
- Pour des erreurs relatives (prix s'étalant de 10 à 10 000), préférez `MAPE` ou travailler sur `log(y)`.

## E.3 Validation croisée (*Cross-validation*)

Un seul découpage train/test peut être trompeur (chance ou malchance dans la répartition). La **validation croisée K-fold** répète l'évaluation sur `K` découpages différents.

```
K=5 :   Pli 1 : [TEST][ train ][ train ][ train ][ train ]
        Pli 2 : [ train ][TEST][ train ][ train ][ train ]
        ...                                                  → 5 scores → moyenne ± écart-type
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0), X, y, cv=skf, scoring="f1")
print(scores.round(3), "→", round(scores.mean(), 3), "±", round(scores.std(), 3))
```

Interprétation : la **moyenne** estime la performance attendue ; l'**écart-type** mesure la stabilité. Un écart-type élevé indique un modèle ou un jeu de données instable.

Variantes :

| Variante | Quand l'utiliser |
|---|---|
| `KFold` | régression |
| `StratifiedKFold` | classification (préserve les proportions) |
| `GroupKFold` | observations groupées (patients, clients) — évite la fuite |
| `TimeSeriesSplit` | séries temporelles (toujours entraîner sur le passé, tester sur le futur) |
| `LeaveOneOut` | très petits jeux (coûteux) |
| `RepeatedStratifiedKFold` | résultats plus stables |

`cross_validate` calcule plusieurs métriques à la fois :

```python
res = cross_validate(RandomForestClassifier(n_estimators=100, random_state=0), X, y, cv=skf,
                     scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True)
print(pd.DataFrame(res).mean().round(3))
```

Comparer des modèles proprement : **mêmes plis** pour tous.

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
modeles = {
    "logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "svm": make_pipeline(StandardScaler(), SVC()),
    "forêt": RandomForestClassifier(n_estimators=200, random_state=0),
}
for nom, mod in modeles.items():
    s = cross_val_score(mod, X, y, cv=skf, scoring="roc_auc")
    print(f"{nom:<11} AUC = {s.mean():.3f} ± {s.std():.3f}")
```

Si deux modèles sont dans la marge d'un écart-type, **leur différence n'est pas significative** : préférez le plus simple.

## E.4 Overfitting et Underfitting

| | **Underfitting** (sous-apprentissage) | **Bon ajustement** | **Overfitting** (sur-apprentissage) |
|---|---|---|---|
| Idée | modèle trop simple | équilibre | modèle qui **mémorise** le bruit |
| Erreur train | **élevée** | faible | **très faible** |
| Erreur test | **élevée** | faible | **élevée** |
| Écart train/test | faible | faible | **grand** |
| Remède | modèle plus complexe, plus de features | — | plus de données, régularisation, modèle plus simple |

```
Erreur
  │\                        test
  │ \                    ╱───────  ← overfitting : le test remonte
  │  \                ╱─╯
  │   ╲_____________╱
  │    ╲___________________  train ← continue à baisser
  └──────────────────────────── complexité du modèle
   sous-apprentissage │ optimal │ sur-apprentissage
```

### Diagnostic : courbe de validation

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

profondeurs = np.arange(1, 16)
train_sc, val_sc = validation_curve(DecisionTreeClassifier(random_state=0), X, y,
                                    param_name="max_depth", param_range=profondeurs, cv=5)
resume = pd.DataFrame({"profondeur": profondeurs, "train": train_sc.mean(axis=1).round(3),
                       "validation": val_sc.mean(axis=1).round(3)})
print(resume.to_string(index=False))
# train → 1.0 ; validation plafonne puis stagne/baisse : l'écart révèle l'overfitting
```

### Diagnostic : courbe d'apprentissage (learning curve)

Performance en fonction de la **taille du train** : si train et validation convergent vers un score **bas** → biais fort (underfitting : plus de données n'aidera pas, il faut un modèle plus riche). Si un grand écart persiste → variance forte (overfitting : plus de données aide).

```python
from sklearn.model_selection import learning_curve
tailles, tr, va = learning_curve(RandomForestClassifier(n_estimators=100, random_state=0), X, y,
                                 train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
print(pd.DataFrame({"n_train": tailles, "train": tr.mean(axis=1).round(3), "valid": va.mean(axis=1).round(3)}))
```

### Remèdes contre l'overfitting

1. Plus de données.
2. Modèle plus simple (profondeur d'arbre, degré polynomial, moins de features).
3. **Régularisation** (Ridge, Lasso, `C` de la SVM, `min_samples_leaf`).
4. Validation croisée pour choisir la complexité.
5. Ensembles (bagging).
6. Arrêt précoce (réseaux de neurones, boosting).
7. Suppression des features bruitées.

## E.5 Compromis biais-variance (*Bias-Variance Tradeoff*)

L'erreur de généralisation d'un modèle se décompose en :

`Erreur = Biais² + Variance + Bruit irréductible`

| Composante | Signification | Cause typique |
|---|---|---|
| **Biais** | erreur due à des hypothèses trop simplistes : le modèle rate systématiquement la cible | modèle trop simple |
| **Variance** | sensibilité du modèle aux **particularités du jeu d'entraînement** : un autre échantillon donnerait un modèle très différent | modèle trop complexe |
| **Bruit** | variabilité intrinsèque des données | irréductible |

Analogie : tir à l'arc. Biais fort = toutes les flèches loin du centre, groupées ; variance forte = flèches dispersées autour du centre ; l'idéal = groupées **et** centrées.

| Modèle | Biais | Variance |
|---|---|---|
| Régression linéaire | élevé | faible |
| Arbre profond | faible | élevée |
| Forêt aléatoire | faible | moyenne (réduite par moyennage) |
| K-NN `k=1` | faible | élevée |
| K-NN `k` grand | élevé | faible |

> 💡 Augmenter la complexité **réduit le biais mais augmente la variance**. L'objectif est le point de rencontre. C'est le fondement du choix d'hyperparamètres.

Simulation :

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
vraie = lambda x: np.sin(1.5 * x)
x_grille = np.linspace(0.3, 3.7, 50)                        # on évalue à l'intérieur de la plage d'entraînement
for degre in [1, 4, 9]:
    predictions = []
    for essai in range(100):                              # 100 jeux d'entraînement différents
        xs = rng.uniform(0, 4, 30); ys = vraie(xs) + rng.normal(0, 0.3, 30)
        mod = make_pipeline(PolynomialFeatures(degre), StandardScaler(), LinearRegression()).fit(xs[:, None], ys)
        predictions.append(mod.predict(x_grille[:, None]))
    predictions = np.array(predictions)
    biais2 = ((predictions.mean(axis=0) - vraie(x_grille)) ** 2).mean()
    variance = predictions.var(axis=0).mean()
    print(f"degré {degre:<2} biais²={biais2:.3f} variance={variance:.3f}")
# degré 1 : biais élevé / variance faible ; degré 9 : biais ≈ 0 mais variance 10 à 100 fois plus grande qu'au degré 4 (selon les tirages)
```

## E.6 Optimisation des hyperparamètres

Rappel : les **hyperparamètres** (profondeur, `k`, `C`, `alpha`…) sont fixés **avant** l'entraînement. On les cherche par validation croisée sur le **train**, puis on évalue **une fois** sur le test.

### Grid Search (recherche en grille)

Teste **toutes** les combinaisons d'une grille.

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC())
grille = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1], "svc__kernel": ["rbf"]}
gs = GridSearchCV(pipe, grille, cv=5, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
print("Meilleurs paramètres :", gs.best_params_)
print("Meilleur score CV    :", round(gs.best_score_, 3))
print("Score sur le TEST    :", round(gs.score(X_te, y_te), 3))        # évaluation finale unique
cv_res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
print(cv_res.sort_values("rank_test_score").head(5))
```

Coût : le nombre d'ajustements = (nombre de combinaisons) × `cv`. 4×4×5 = 80 ajustements ici.

### Random Search (recherche aléatoire)

Tire **au hasard** `n_iter` combinaisons dans des distributions. Plus efficace quand il y a beaucoup d'hyperparamètres, car tous ne comptent pas autant (Bergstra & Bengio, 2012).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

espace = {"n_estimators": randint(100, 500), "max_depth": randint(3, 20),
          "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", "log2", 0.5]}
rs = RandomizedSearchCV(RandomForestClassifier(random_state=0), espace, n_iter=20, cv=5,
                        scoring="f1", random_state=0, n_jobs=-1).fit(X_tr, y_tr)
print(rs.best_params_, round(rs.best_score_, 3))
```

| | Grid Search | Random Search |
|---|---|---|
| Couverture | exhaustive sur la grille | échantillonnage |
| Coût | explose avec le nombre d'hyperparamètres | maîtrisé (`n_iter`) |
| Quand | 1–3 hyperparamètres, grille petite | beaucoup d'hyperparamètres, plages larges |
| Variantes avancées | `HalvingGridSearchCV` | optimisation bayésienne (Optuna) |

⚠️ **Pièges**
- **Ne jamais** régler les hyperparamètres sur le jeu de test : il devient un jeu d'entraînement déguisé. On règle en CV sur le train.
- Mettre le prétraitement **dans le pipeline** pour qu'il soit refit à chaque pli.
- Une différence de 0,2 point entre deux configurations est souvent du bruit : regardez `std_test_score`.
- Rester sur des plages raisonnables ; échelle **logarithmique** pour `C`, `gamma`, `alpha`.

## E.7 Récapitulatif : comment interpréter vos résultats

| Observation | Diagnostic probable | Action |
|---|---|---|
| Train 99 %, test 70 % | overfitting | régulariser, simplifier, plus de données |
| Train 65 %, test 63 % | underfitting | modèle plus riche, meilleures features |
| Accuracy 98 %, rappel de la classe rare 5 % | classes déséquilibrées | F1/rappel, `class_weight`, rééchantillonnage, seuil |
| CV 0,85 ± 0,10 | instable | plus de données, modèle plus simple, `Repeated` CV |
| Test bien meilleur que la CV | chance, ou fuite | enquêter |
| Score trop beau pour être vrai | **data leakage** | revoir features et pipeline |
| Modèle complexe ≈ baseline | peu de signal | revoir les données/features |
| R² négatif sur test | pire que la moyenne | modèle inadapté ou overfit |

## Exercices E

1. Un test détecte une maladie : VP=40, FN=10, FP=30, VN=920. Calculez à la main accuracy, précision, rappel, F1, spécificité. Quel est le problème de l'accuracy ici ?
2. Dans quel cas privilégier le rappel ? la précision ? Donnez un exemple pour chaque.
3. Pourquoi RMSE ≥ MAE ? Que signifie un RMSE très supérieur à la MAE ?
4. Sur `load_wine`, comparez un arbre de profondeur 1, 3, 10, illimitée par validation croisée 5 plis. Tracez train vs validation.
5. Lancez un `GridSearchCV` pour un K-NN (k de 1 à 25, `weights`) sur les iris. Quel est le meilleur `k` ? Le score sur le test est-il plus bas que le score CV ? Pourquoi est-ce normal ?
6. Expliquez en une phrase le compromis biais-variance pour la profondeur d'un arbre.

---

# PARTIE F — Construire un projet de Machine Learning

## F.1 Le workflow en 12 étapes

```
1 Problème → 2 Données → 3 Exploration → 4 Nettoyage → 5 Features → 6 Split
                                                                      │
12 Sauvegarde ← 11 Prédiction ← 10 Amélioration ← 9 Évaluation ← 8 Entraînement ← 7 Choix du modèle
```

C'est un processus **itératif** : les résultats de l'évaluation renvoient souvent à l'exploration et aux features.

| # | Étape | Question centrale | Livrable |
|---|---|---|---|
| 1 | Définition du problème | Que prédire ? Pour qui ? Quel gain ? Quelle métrique ? | cahier des charges d'une page |
| 2 | Collecte | Les données sont-elles accessibles, légales, représentatives ? | dataset brut (intouché) |
| 3 | Exploration | Que contient-il ? Quels pièges ? | notebook d'EDA |
| 4 | Nettoyage | Comment corriger types, doublons, NA, outliers ? | règles de nettoyage |
| 5 | Feature engineering | Quelles variables ajouter/retirer ? | pipeline de features |
| 6 | Séparation | Comment éviter toute fuite ? | train / test (+ CV) |
| 7 | Choix du modèle | Baseline + candidats | liste de modèles |
| 8 | Entraînement | Quels paramètres appris ? | modèle ajusté |
| 9 | Évaluation | Bon selon la **bonne** métrique ? | rapport de métriques |
| 10 | Amélioration | Où sont les erreurs ? Quels réglages ? | modèle optimisé |
| 11 | Prédiction | Comment l'utiliser sur de nouvelles données ? | fonction/API de prédiction |
| 12 | Sauvegarde | Comment le réutiliser, le versionner ? | fichier modèle + doc |

### Détail et pièges de chaque étape

**1. Définir le problème.** Traduisez le besoin métier en tâche ML : « réduire les impayés » → classification binaire du risque de défaut. Définissez le **critère de succès** (ex. rappel ≥ 70 % avec précision ≥ 40 %) et la **baseline** actuelle. Posez-vous la question : *un modèle est-il vraiment nécessaire ?* Vérifiez qu'au moment de la prédiction vous disposez de toutes les variables.

**2. Collecter.** Documentez la provenance, la période, la définition de chaque colonne. Ne modifiez jamais le fichier brut.

**3. Explorer.** Distribution de la cible, valeurs manquantes, corrélations, outliers, groupes atypiques.

**4. Nettoyer.** Types, doublons, cohérence, NA (stratégie documentée).

**5. Features.** Ratios, dates, regroupements ; supprimez les identifiants (un `id_client` n'est pas une variable prédictive).

**6. Séparer.** Split stratifié, `random_state` fixé ; mettre le jeu de test **de côté** jusqu'à la fin.

**7. Choisir.** Toujours une baseline (`DummyClassifier`, régression logistique) puis des modèles plus riches.

**8. Entraîner.** Via un `Pipeline` pour éviter les fuites.

**9. Évaluer.** Validation croisée sur le train ; métrique alignée sur le coût métier ; matrice de confusion ; analyse des erreurs.

**10. Améliorer.** Dans l'ordre d'efficacité habituel : (1) meilleures données/features ; (2) autre modèle ; (3) réglage d'hyperparamètres ; (4) ensembles.

**11. Prédire.** Les nouvelles données doivent subir **exactement** les mêmes transformations.

**12. Sauvegarder.** Le pipeline complet (pas seulement le modèle), avec les versions des bibliothèques.

## F.2 Projet guidé de bout en bout : prédire la résiliation d'abonnés (*churn*)

**Contexte.** Un opérateur télécom veut identifier les abonnés qui risquent de résilier dans les 30 jours pour leur proposer une offre de fidélisation.

### Étape 1 — Définition du problème

- **Tâche** : classification binaire (`resilie` = 1 / 0).
- **Coût des erreurs** : proposer une offre à un client fidèle (FP) coûte un petit rabais ; **laisser partir un client** (FN) coûte beaucoup plus. On privilégiera le **rappel** tout en gardant une précision acceptable → métrique de sélection : **F1** (et AUC pour comparer), puis réglage du seuil.
- **Baseline** : prédire « personne ne résilie » (accuracy ≈ taux de fidèles).
- **Disponibilité des variables à la prédiction** : toutes les variables ci-dessous sont connues avant la résiliation.

### Étape 2 — Collecte (jeu réaliste généré)

Nous générons un jeu réaliste dont la probabilité de résiliation dépend de facteurs plausibles (ancienneté, type de contrat, appels au support, facture élevée…). Les données sont **sauvegardées en CSV brut** et ne seront plus modifiées.

```python
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(2025)
n = 5000

contrat = rng.choice(["Mensuel", "Annuel", "Biannuel"], n, p=[0.55, 0.30, 0.15])
anciennete = np.where(contrat == "Mensuel", rng.integers(1, 40, n), rng.integers(6, 72, n))
forfait = rng.choice(["Basique", "Standard", "Premium"], n, p=[0.35, 0.45, 0.20])
prix_forfait = pd.Series(forfait).map({"Basique": 5000, "Standard": 9000, "Premium": 15000}).to_numpy()
data_go = np.clip(rng.gamma(2.0, 3.0, n) * (1 + 0.5 * (forfait == "Premium")), 0, 80).round(1)
appels_support = rng.poisson(0.6 + 0.8 * (contrat == "Mensuel"), n)
retards_paiement = rng.poisson(0.3, n)
facture = (prix_forfait + rng.normal(0, 800, n) + data_go * 20).round(-2)
paiement = rng.choice(["Mobile Money", "Carte", "Espèces", "Prélèvement"], n, p=[0.5, 0.15, 0.25, 0.10])
age = rng.integers(18, 70, n)

logit = (-1.0
         - 0.045 * anciennete
         + 1.1 * (contrat == "Mensuel") - 0.9 * (contrat == "Biannuel")
         + 0.35 * appels_support
         + 0.4 * retards_paiement
         + 0.00008 * (facture - 9000)
         + 0.4 * (paiement == "Espèces")
         - 0.3 * (paiement == "Prélèvement")
         - 0.008 * (age - 40)
         + rng.normal(0, 0.6, n))
resilie = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

abonnes = pd.DataFrame({
    "id_client": np.arange(100000, 100000 + n), "age": age, "anciennete_mois": anciennete,
    "contrat": contrat, "forfait": forfait, "data_go": data_go, "facture_mensuelle": facture,
    "appels_support": appels_support, "retards_paiement": retards_paiement,
    "mode_paiement": paiement, "resilie": resilie,
})

# Imperfections réalistes
abonnes.loc[rng.choice(n, 150, replace=False), "age"] = np.nan
abonnes.loc[rng.choice(n, 100, replace=False), "mode_paiement"] = np.nan
abonnes.loc[rng.choice(n, 20, replace=False), "facture_mensuelle"] = -1          # valeur sentinelle absurde
abonnes.loc[rng.choice(n, 10, replace=False), "data_go"] = 999                   # erreur de saisie

dossier = Path("data"); dossier.mkdir(exist_ok=True)
abonnes.to_csv(dossier / "abonnes_brut.csv", index=False)
print(abonnes.shape)
```

### Étape 3 — Exploration

```python
df = pd.read_csv("data/abonnes_brut.csv")
df.info()
print(df.describe().round(1).T)
print("Taux de résiliation :", df["resilie"].mean().round(3))
print(df.isna().sum()[lambda s: s > 0])
print("Doublons :", df.duplicated().sum(), "| id dupliqués :", df["id_client"].duplicated().sum())

# Lien entre variables et cible
print(df.groupby("contrat")["resilie"].agg(["mean", "size"]).round(3))
print(df.groupby("appels_support")["resilie"].mean().round(3))
print(df.groupby(pd.cut(df["anciennete_mois"], [0, 6, 12, 24, 48, 100]), observed=True)["resilie"].mean().round(3))
```

Constats attendus : environ 30 % de résiliations ; le contrat **mensuel** résilie bien davantage que les contrats longs ; les clients de moins de 6 mois partent beaucoup plus ; plus d'appels au support = plus de départs ; des valeurs aberrantes existent (`facture = -1`, `data_go = 999`) et des valeurs manquantes (âge, mode de paiement).

### Étape 4 — Nettoyage

```python
df = df.drop_duplicates(subset="id_client").copy()
df = df.drop(columns="id_client")                      # un identifiant n'est pas une variable prédictive

df.loc[df["facture_mensuelle"] < 0, "facture_mensuelle"] = np.nan      # sentinelle → manquant
df.loc[df["data_go"] > 100, "data_go"] = np.nan                        # impossible → manquant
print(df.isna().sum()[lambda s: s > 0])
```

Les valeurs manquantes seront **imputées dans le pipeline** (après le split).

### Étape 5 — Feature engineering

```python
df["facture_par_go"] = df["facture_mensuelle"] / (df["data_go"] + 1)
df["client_recent"] = (df["anciennete_mois"] <= 6).astype(int)
df["problemes"] = df["appels_support"] + df["retards_paiement"]
```

### Étape 6 — Séparation

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns="resilie")
y = df["resilie"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape, y_train.mean().round(3), y_test.mean().round(3))
```

**À partir d'ici, `X_test` est mis de côté** jusqu'à l'évaluation finale.

### Étape 7 — Choix du modèle et construction du pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ["age", "anciennete_mois", "data_go", "facture_mensuelle", "appels_support",
            "retards_paiement", "facture_par_go", "client_recent", "problemes"]
cat_cols = ["contrat", "forfait", "mode_paiement"]

prepro = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
```

Candidats : baseline (`DummyClassifier`), régression logistique, forêt aléatoire, gradient boosting.

### Étape 8 — Entraînement et comparaison par validation croisée

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

candidats = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistique": LogisticRegression(max_iter=1000),
    "forêt": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0),
    "boosting": HistGradientBoostingClassifier(random_state=0),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
lignes = []
for nom, clf in candidats.items():
    pipe = Pipeline([("prepro", prepro), ("clf", clf)])
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    lignes.append({"modèle": nom, **{m: r[f"test_{m}"].mean().round(3) for m in ["accuracy", "f1", "roc_auc"]}})
print(pd.DataFrame(lignes).to_string(index=False))
```

🔍 **Lire le tableau** : la baseline « personne ne résilie » atteint une accuracy ≈ 0,70 (proportion de fidèles) mais un F1 = 0 : elle ne détecte aucun départ, d'où l'inutilité de l'accuracy seule ici. La régression logistique obtient F1 ≈ 0,52 et AUC ≈ 0,80, et **ni la forêt ni le boosting ne la battent** (leurs scores sont même un peu inférieurs) : la relation est essentiellement linéaire dans le logit, la complexité supplémentaire n'apporte que du bruit. **Quand un modèle simple est aussi bon ou meilleur, gardez-le** (plus facile à expliquer, à maintenir et à surveiller).

### Étape 9 — Évaluation sur le jeu de test (une seule fois)

On retient la **régression logistique** (simple, explicable). On entraîne sur tout le train et on évalue sur le test.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

final = Pipeline([("prepro", prepro), ("clf", LogisticRegression(max_iter=1000))])
final.fit(X_train, y_train)
pred = final.predict(X_test)
proba = final.predict_proba(X_test)[:, 1]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
print("AUC :", round(roc_auc_score(y_test, proba), 3))
```

### Étape 10 — Amélioration

**a) Optimiser les hyperparamètres** (dans la CV du train).

```python
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(final, {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
                  cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
print(gs.best_params_, round(gs.best_score_, 3))
meilleur = gs.best_estimator_
```

**b) Ajuster le seuil de décision** selon le coût métier. Hypothèse : retenir un client coûte 2 000 FCFA (offre) ; perdre un client coûte 30 000 FCFA ; une offre acceptée par un client qui serait parti le sauve avec une probabilité de 40 %.

```python
proba_cv = meilleur.predict_proba(X_test)[:, 1]
cout_offre, valeur_client, taux_succes = 2_000, 30_000, 0.40

def gain_total(seuil):
    cible = proba_cv >= seuil
    vrais_partants = (y_test.to_numpy() == 1) & cible
    gain = vrais_partants.sum() * taux_succes * valeur_client - cible.sum() * cout_offre
    return gain

seuils = np.arange(0.1, 0.9, 0.05)
gains = [gain_total(s) for s in seuils]
meilleur_seuil = seuils[int(np.argmax(gains))]
print(f"seuil optimal ≈ {meilleur_seuil:.2f}, gain ≈ {max(gains):,.0f} FCFA")
```

Le seuil optimal découle de l'économie du problème, pas de la valeur « 0,5 » par défaut.

**c) Interpréter le modèle** : quels facteurs poussent à la résiliation ?

```python
noms = meilleur[:-1].get_feature_names_out()
coefs = pd.Series(meilleur[-1].coef_[0], index=noms).sort_values()
print(coefs.round(2))
```

Lecture : coefficient positif = ↑ risque de départ (ex. `contrat_Mensuel`, `appels_support`), négatif = fidélise (`anciennete_mois`, `contrat_Biannuel`). Cette lecture permet des **actions** : proposer un contrat annuel remisé aux clients mensuels récents, traiter vite les clients qui appellent le support.

**d) Analyser les erreurs** : quels clients sont mal classés ? Y a-t-il un segment où le modèle est moins bon ?

```python
res = X_test.copy(); res["vrai"] = y_test.to_numpy(); res["proba"] = proba_cv
res["erreur"] = (res["proba"] >= meilleur_seuil).astype(int) != res["vrai"]
print(res.groupby("contrat")["erreur"].mean().round(3))
```

### Étape 11 — Prédiction sur de nouveaux clients

```python
nouveaux = pd.DataFrame({
    "age": [28, 52], "anciennete_mois": [3, 48], "contrat": ["Mensuel", "Annuel"],
    "forfait": ["Basique", "Premium"], "data_go": [4.0, 22.0], "facture_mensuelle": [5400, 15800],
    "appels_support": [3, 0], "retards_paiement": [1, 0], "mode_paiement": ["Espèces", "Prélèvement"],
})
nouveaux["facture_par_go"] = nouveaux["facture_mensuelle"] / (nouveaux["data_go"] + 1)
nouveaux["client_recent"] = (nouveaux["anciennete_mois"] <= 6).astype(int)
nouveaux["problemes"] = nouveaux["appels_support"] + nouveaux["retards_paiement"]

p = meilleur.predict_proba(nouveaux)[:, 1]
nouveaux["risque_resiliation"] = p.round(3)
nouveaux["action"] = np.where(p >= meilleur_seuil, "Proposer offre", "Aucune action")
print(nouveaux[["contrat", "anciennete_mois", "risque_resiliation", "action"]])
```

Le premier client (nouveau, mensuel, nombreux appels) présente un risque élevé ; le second (ancien, annuel, prélèvement) un risque faible.

### Étape 12 — Sauvegarde et réutilisation

On sauvegarde le **pipeline complet** (prétraitement + modèle) afin que les nouvelles données subissent exactement les mêmes transformations.

```python
import joblib, json, sklearn
Path("modeles").mkdir(exist_ok=True)
joblib.dump(meilleur, "modeles/churn_pipeline.joblib")
meta = {"sklearn": sklearn.__version__, "seuil": float(meilleur_seuil),
        "features": list(X_train.columns), "metrique_cv_f1": float(gs.best_score_)}
Path("modeles/churn_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

# --- plus tard, dans un autre script ---
modele = joblib.load("modeles/churn_pipeline.joblib")
print(modele.predict_proba(nouveaux[meta["features"]])[:, 1].round(3))
```

⚠️ **Sécurité** : ne chargez jamais (`joblib`/`pickle`) un fichier modèle venant d'une source non fiable : le chargement peut exécuter du code arbitraire. **Versions** : un modèle sauvegardé avec une version de scikit-learn peut ne pas se recharger avec une autre ; notez la version.

### Ce que ce projet vous apprend

1. Cadrer le problème avec la **métrique métier** avant de modéliser.
2. Le **pipeline** garantit l'absence de fuite et la reproductibilité.
3. Une **baseline** et des modèles simples fixent la barre.
4. Le **seuil** se choisit par l'économie, pas par défaut.
5. Interpréter pour **agir**, pas seulement prédire.
6. Sauvegarder le **pipeline entier**.

### Après le déploiement

- **Surveiller la dérive** (*drift*) : le comportement des clients évolue, les performances se dégradent. Comparer régulièrement la distribution des features et la performance réelle.
- **Réentraîner** périodiquement.
- Mesurer l'**impact causal** (test A/B : groupe traité vs groupe témoin), car prédire le départ ne prouve pas que l'offre le **cause** d'être évité.

## F.3 Modèle de structure de projet

```
mon_projet/
├── data/
│   ├── brut/            # données d'origine, jamais modifiées
│   └── traite/          # résultats du nettoyage
├── notebooks/           # exploration (01_eda.ipynb, 02_modeles.ipynb)
├── src/                 # code réutilisable (prepro.py, train.py, predict.py)
├── modeles/             # pipelines sauvegardés + métadonnées
├── rapports/            # figures, résultats
├── requirements.txt     # versions exactes des bibliothèques
└── README.md            # problème, métrique, résultats, comment relancer
```

## Exercices F

1. Reprenez le projet churn avec `HistGradientBoostingClassifier` et comparez avec la logistique. Le gain justifie-t-il la complexité ?
2. Que se passe-t-il si vous incluez par erreur une colonne `date_resiliation` comme feature ? Pourquoi est-ce une fuite ?
3. Modifiez les coûts (`cout_offre = 10_000`). Comment évolue le seuil optimal ? Pourquoi ?
4. Écrivez une fonction `preparer(df)` qui applique les étapes 4 et 5 et réutilisez-la pour les nouveaux clients.
5. Rédigez le `README.md` de ce projet en 10 lignes.

---

# PARTIE G — Aller plus loin

> Cette partie est une **carte du territoire** pour la suite de votre formation. Les fondamentaux (A–F) restent prioritaires : un excellent praticien du ML classique progresse vite en Deep Learning ; l'inverse est faux.

## G.1 Ensembles de modèles

Combiner plusieurs modèles pour obtenir un modèle meilleur que chacun.

| Famille | Principe | Réduit | Exemple |
|---|---|---|---|
| **Bagging** | modèles indépendants sur des échantillons bootstrap, moyennés | variance | Random Forest |
| **Boosting** | modèles **séquentiels**, chacun corrige les erreurs des précédents | biais (et variance) | AdaBoost, Gradient Boosting, XGBoost |
| **Stacking** | un méta-modèle apprend à combiner les prédictions de modèles variés | les deux | `StackingClassifier` |
| **Voting** | vote/moyenne simple | variance | `VotingClassifier` |

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
base = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=0))]
vote = VotingClassifier(base, voting="soft")
stack = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000), cv=5)
for nom, m in [("vote", vote), ("stacking", stack)]:
    print(nom, cross_val_score(m, X, y, cv=5).mean().round(3))
```

Les ensembles gagnent quand les modèles sont **performants et différents** (erreurs peu corrélées).

## G.2 Gradient Boosting

**Idée** : construire des arbres **un par un**, chaque nouvel arbre apprenant les **erreurs résiduelles** (le gradient de la perte) du modèle courant.

`F_m(x) = F_{m−1}(x) + η · h_m(x)`

où `h_m` est un petit arbre ajusté sur les résidus et `η` le taux d'apprentissage (*learning rate*) : c'est une descente de gradient **dans l'espace des fonctions**.

```
prédiction = arbre₁ + η·arbre₂ + η·arbre₃ + … (chaque arbre corrige les erreurs restantes)
```

| Hyperparamètre | Effet |
|---|---|
| `n_estimators` | nombre d'arbres (plus = plus de capacité, risque d'overfit) |
| `learning_rate` | pas : petit = plus d'arbres nécessaires mais meilleure généralisation |
| `max_depth` | arbres peu profonds (3–6) |
| `subsample` | fraction de lignes par arbre (aléa → régularisation) |
| `early_stopping` | arrêter quand la validation ne s'améliore plus |

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0).fit(X_tr, y_tr)
hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, early_stopping=True, random_state=0).fit(X_tr, y_tr)
print(round(gb.score(X_te, y_te), 3), round(hgb.score(X_te, y_te), 3))
```

`HistGradientBoosting` (histogrammes) est beaucoup plus rapide sur gros jeux et gère nativement les valeurs manquantes.

## G.3 XGBoost (et LightGBM, CatBoost)

**XGBoost** (*eXtreme Gradient Boosting*) est une implémentation optimisée du gradient boosting : régularisation L1/L2 sur les arbres, gestion des manquants, parallélisme, arrêt précoce. Il domine les compétitions sur **données tabulaires**. Alternatives : **LightGBM** (très rapide), **CatBoost** (excellent sur les catégories).

```python
# pip install xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                    early_stopping_rounds=30, random_state=0)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=0, stratify=y_tr)
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print("arbres utilisés :", xgb.best_iteration + 1, "| test :", round(xgb.score(X_te, y_te), 3))
```

Conseils : commencer avec `learning_rate` 0,05–0,1 et `early_stopping` ; profondeur 3–6 ; régler `subsample`/`colsample_bytree` ; garder un jeu de validation distinct du test.

## G.4 Introduction aux réseaux de neurones

Un **neurone artificiel** calcule `a = f(w·x + b)` : combinaison linéaire des entrées suivie d'une **fonction d'activation** non linéaire `f` (ReLU `max(0,z)`, sigmoïde, tanh). Un **réseau** empile des couches de neurones.

```
entrées → [couche cachée 1] → [couche cachée 2] → sortie
  x₁ ●──╲ ╱─● ──╲ ╱─● ──╲
  x₂ ●────╳──●────╳──●────● ŷ
  x₃ ●──╱ ╲─● ──╱ ╲─● ──╱
```

Sans non-linéarité, plusieurs couches équivaudraient à une seule transformation linéaire : l'**activation** apporte la puissance d'expression.

**Entraînement** :
1. **Propagation avant** (*forward*) : calcul de la sortie ;
2. **Perte** (ex. entropie croisée) ;
3. **Rétropropagation** (*backpropagation*) : calcul des gradients de la perte par rapport à chaque poids via la **règle de la chaîne** (B.6) ;
4. **Mise à jour** par descente de gradient (SGD, Adam) ;
5. répétition sur des *epochs* (passes complètes) par mini-lots.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

Xd, yd = load_digits(return_X_y=True)                      # images 8×8 de chiffres
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd / 16.0, yd, test_size=0.25, random_state=0, stratify=yd)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=0).fit(Xd_tr, yd_tr)
print("accuracy (chiffres) :", round(mlp.score(Xd_te, yd_te), 3))
```

Pour des réseaux de taille réelle, utilisez TensorFlow/Keras ou PyTorch (G.6).

## G.5 Deep Learning

Réseaux **profonds** (nombreuses couches) qui apprennent eux-mêmes des **représentations** hiérarchiques (pixels → bords → formes → objets).

| Architecture | Données | Idée | Applications |
|---|---|---|---|
| **MLP** (perceptron multicouche) | tabulaire | couches entièrement connectées | prédiction |
| **CNN** (convolutif) | images | filtres locaux partageant leurs poids | classification d'images, médical |
| **RNN / LSTM** | séquences | mémoire d'état | séries temporelles (ancien) |
| **Transformer** | texte, image, audio | **attention** : chaque élément pondère les autres | modèles de langage, traduction |
| **Autoencodeur / GAN / Diffusion** | tous | génération, compression | images générées |

Quand utiliser le Deep Learning ? Beaucoup de données (≥ dizaines de milliers), données non structurées (images, sons, texte), puissance de calcul (GPU). Sur des tableaux de quelques milliers de lignes : arbres et boosting font généralement aussi bien ou mieux.

Techniques propres au Deep Learning : *dropout*, *batch normalization*, *early stopping*, *data augmentation*, **apprentissage par transfert** (réutiliser un modèle pré-entraîné et l'affiner : indispensable en pratique).

## G.6 TensorFlow et PyTorch

| | **PyTorch** | **TensorFlow / Keras** |
|---|---|---|
| Style | Python natif, graphe dynamique | Keras haut niveau |
| Populaire en | recherche, modèles récents | production, mobile (TFLite) |
| Courbe d'apprentissage | boucle d'entraînement explicite | très accessible (`model.fit`) |
| Écosystème | Hugging Face, Lightning | TFX, TF Serving |

Les deux fournissent : des tenseurs (comme NumPy, mais GPU), la **différentiation automatique** (calcul automatique des gradients), des couches et optimiseurs prêts à l'emploi.

Exemple PyTorch minimal (la régression linéaire de B.8, mais avec différentiation automatique) :

```python
# pip install torch
import torch
x = torch.rand(100, 1) * 10
y = 3 * x + 5 + torch.randn(100, 1) * 2
modele = torch.nn.Linear(1, 1)
opt = torch.optim.SGD(modele.parameters(), lr=0.01)
for _ in range(2000):
    perte = torch.nn.functional.mse_loss(modele(x), y)
    opt.zero_grad(); perte.backward(); opt.step()          # backward() = rétropropagation automatique
print(modele.weight.item(), modele.bias.item())            # ≈ 3, ≈ 5
```

Exemple Keras :

```python
# pip install tensorflow
from tensorflow import keras
m = keras.Sequential([keras.layers.Input((20,)), keras.layers.Dense(32, activation="relu"),
                      keras.layers.Dense(1, activation="sigmoid")])
m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# m.fit(X_train, y_train, epochs=20, validation_split=0.2)
```

## G.7 NLP (*Natural Language Processing*)

Traitement automatique du **langage**. Étapes classiques :

1. **Prétraitement** : minuscules, tokenisation, suppression des mots vides, lemmatisation.
2. **Représentation** : sac de mots (`CountVectorizer`), **TF-IDF** (pondère les mots fréquents dans un document mais rares dans le corpus), puis **embeddings** (vecteurs denses où les mots proches par le sens sont proches dans l'espace).
3. **Modèle** : logistique/Naive Bayes sur TF-IDF (baseline très forte), puis Transformers.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

avis = ["service excellent, très satisfait", "produit défectueux, déçu", "livraison rapide et conforme",
        "arnaque, ne jamais racheter", "super qualité, je recommande", "très mauvaise expérience, remboursement refusé"]
sent = [1, 0, 1, 0, 1, 0]
nlp = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression()).fit(avis, sent)
print(nlp.predict(["qualité excellente", "expérience mauvaise, déçu"]))       # [1 0]
```

Tâches : classification de sentiment, extraction d'entités, traduction, résumé, questions-réponses, génération. Les **modèles de langage** de type Transformer (pré-entraînés sur de très grands corpus) sont aujourd'hui l'état de l'art ; on les utilise via des bibliothèques comme Hugging Face `transformers` ou via des API.

## G.8 Computer Vision

Faire « voir » les machines. Une image est un tableau NumPy `(hauteur, largeur, canaux)`.

| Tâche | Sortie | Exemple |
|---|---|---|
| Classification | une étiquette | chat/chien |
| Détection d'objets | boîtes + étiquettes | voitures sur la route |
| Segmentation | étiquette par pixel | tumeur sur une IRM |
| Reconnaissance faciale, OCR, estimation de pose | … | lecture de plaques |

Outils : **OpenCV** (traitement), **torchvision / Keras** (modèles), CNN pré-entraînés (ResNet, EfficientNet) adaptés par apprentissage par transfert, puis Vision Transformers.

```python
from sklearn.datasets import load_digits
img = load_digits().images[0]
print(img.shape, img.min(), img.max())          # (8, 8) 0.0 15.0 : une image = une matrice
```

## G.9 MLOps

**MLOps** (*Machine Learning Operations*) = appliquer les pratiques DevOps au cycle de vie des modèles : passer d'un notebook à un système fiable en production.

| Domaine | Pratique | Outils |
|---|---|---|
| Versionnement | code, **données** et modèles | Git, DVC |
| Suivi d'expériences | enregistrer paramètres/métriques | MLflow, Weights & Biases |
| Pipelines | automatiser préparation → entraînement | scikit-learn Pipelines, Airflow, Prefect |
| Tests | tests de données et de code | pytest, Great Expectations |
| Déploiement | service d'inférence | FastAPI, Docker |
| Surveillance | dérive des données, performance | Evidently, Prometheus |
| Réentraînement | périodique ou déclenché | CI/CD |
| Gouvernance | traçabilité, équité, conformité | cartes de modèle |

## G.10 Déploiement de modèles

Modes de mise en service :

| Mode | Principe | Quand |
|---|---|---|
| **Batch** | prédictions planifiées sur un lot (la nuit) | scores clients, rapports |
| **API temps réel** | service web renvoyant une prédiction | fraude à l'achat, recommandations |
| **Embarqué / edge** | modèle dans l'appli mobile ou l'appareil | hors ligne, latence faible |
| **Streaming** | traitement d'un flux d'événements | capteurs |

Exemple d'API avec FastAPI :

```python
# serveur.py  —  pip install fastapi uvicorn joblib pandas scikit-learn
# lancer :  uvicorn serveur:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI()
modele = joblib.load("modeles/churn_pipeline.joblib")

class Abonne(BaseModel):
    age: float | None = None
    anciennete_mois: int
    contrat: str
    forfait: str
    data_go: float
    facture_mensuelle: float
    appels_support: int
    retards_paiement: int
    mode_paiement: str | None = None

@app.post("/predire")
def predire(a: Abonne):
    d = pd.DataFrame([a.model_dump()])
    d["facture_par_go"] = d["facture_mensuelle"] / (d["data_go"] + 1)
    d["client_recent"] = (d["anciennete_mois"] <= 6).astype(int)
    d["problemes"] = d["appels_support"] + d["retards_paiement"]
    return {"risque": float(modele.predict_proba(d)[:, 1][0])}
```

Points de vigilance : **cohérence des transformations** entre entraînement et production ; **latence** ; **surveillance** ; sécurité des données ; documentation ; mécanisme de **retour arrière**.

## G.11 Feuille de route pour la suite

1. **Consolider** : refaire plusieurs projets complets sur des datasets variés (Kaggle : Titanic, prix des maisons, churn).
2. **Statistiques** : tests d'hypothèses, intervalles de confiance, inférence causale de base.
3. **ML avancé** : gradient boosting (XGBoost/LightGBM), séries temporelles, systèmes de recommandation, détection d'anomalies.
4. **Deep Learning** : PyTorch, CNN, puis Transformers ; apprentissage par transfert.
5. **MLOps** : Git, Docker, API, suivi d'expériences, un projet déployé.
6. **Portfolio** : 3 à 5 projets documentés sur GitHub avec un README clair.

Ressources recommandées : la documentation de scikit-learn (excellente, avec guide utilisateur), *Hands-On Machine Learning* (A. Géron), les cours d'Andrew Ng, fast.ai, Kaggle Learn.

---

# Glossaire

| Terme | Définition |
|---|---|
| **Biais (bias)** | erreur systématique due à un modèle trop simple |
| **Variance** | sensibilité aux fluctuations du jeu d'entraînement |
| **Overfitting** | le modèle mémorise le bruit : bon sur le train, mauvais sur le test |
| **Underfitting** | modèle trop simple : mauvais partout |
| **Feature** | variable d'entrée |
| **Label / target** | variable à prédire |
| **Hyperparamètre** | réglage fixé avant l'apprentissage |
| **Pipeline** | enchaînement d'étapes de transformation et d'un modèle |
| **Data leakage** | fuite d'information du test vers l'entraînement |
| **Cross-validation** | évaluation répétée sur plusieurs découpages |
| **Baseline** | modèle de référence très simple |
| **One-Hot** | encodage d'une catégorie en colonnes binaires |
| **Gradient** | vecteur des dérivées partielles |
| **Learning rate** | taille du pas de la descente de gradient |
| **Epoch** | passe complète sur le jeu d'entraînement |
| **Régularisation** | pénalité contre la complexité |
| **Ensemble** | combinaison de plusieurs modèles |
| **Dérive (drift)** | changement des données au fil du temps |
| **Inférence** | utilisation d'un modèle entraîné pour prédire |
| **Embedding** | représentation vectorielle dense d'un objet (mot, image…) |

---

# Solutions des exercices

## Solutions A

1. (a) régression ; (b) classification binaire ; (c) clustering ; (d) régression ; (e) classification multi-classes (10 chiffres, vision).
2. `n_observations = 1000`, `n_features = 7` (la cible exclue), `X.shape = (1000, 7)`, `y.shape = (1000,)`.
3. Un **paramètre** est appris par l'algorithme (seuil et variable choisis à chaque nœud) ; un **hyperparamètre** est fixé par vous avant (`max_depth`).
4. La règle (TVA = taux × montant) est connue, exacte et stable : apprendre une règle déjà connue serait inutile et moins fiable.

## Solutions B

```python
import numpy as np

# 1
print(np.linalg.norm(np.array([4, 6]) - np.array([1, 2])))     # 5.0  (√(3²+4²))

# 2 : X @ w -> (100,) ; w @ X échoue car (5,) @ (100,5) : dimensions intérieures 5 ≠ 100.

# 3
p_m, se, fp = 0.02, 0.90, 0.10
p_pos = se * p_m + fp * (1 - p_m)
print(round(se * p_m / p_pos, 3))                              # 0.155

# 4 : f'(x) = 2x - 4 = 0 -> x = 2 ; f(2) = 1
x = 0.0
for _ in range(50):
    x -= 0.1 * (2 * x - 4)
print(round(x, 4))                                             # ≈ 2.0

# 5
rng = np.random.default_rng(0)
x = rng.normal(size=500)
print(np.corrcoef(x, 2 * x + rng.normal(0, 0.5, 500))[0, 1].round(2))   # ≈ 0.97
xs = np.linspace(-3, 3, 500)                                            # x symétrique autour de 0
print(np.corrcoef(xs, xs ** 2)[0, 1].round(2))                          # 0.0 : relation parfaite mais non linéaire
```

Pour 5 : la corrélation de Pearson mesure uniquement la relation **linéaire**. Pour `y = x²` avec `x` symétrique autour de 0, les valeurs hautes et basses de `x` donnent toutes deux de grands `y` : les effets se compensent.

## Solutions C

1. L'imputation (moyenne, médiane) utilise des statistiques calculées sur les données. Si on les calcule avant le split, la médiane du test influence le train : le modèle « voit » indirectement le test. On `fit` l'imputeur sur le train uniquement.
2. (a) One-Hot ; (b) ordinal (S<M<L<XL) ; (c) fréquence ou encodage par cible (avec validation croisée interne), ou regroupement des modalités rares.
3. Standardisation nécessaire : **K-NN, SVM, régression logistique** (distances/gradients). Pas nécessaire : **arbre, forêt**.
4. La durée du séjour de **réadmission** n'est connue qu'**après** la réadmission : on ne peut pas la connaître au moment de la prédiction. C'est une fuite temporelle/de cible.
5. Remplacez `StandardScaler()` par `RobustScaler()` et `OneHotEncoder(handle_unknown="ignore")` par `OneHotEncoder(drop="first", handle_unknown="ignore")` dans le `ColumnTransformer` de C.11. L'AUC change peu (± 1 point) ; `RobustScaler` est plus robuste aux outliers de revenu. Remarquez que l'accuracy (≈ 0,78) ne bat pas la baseline « toujours 0 » (≈ 0,79) : c'est l'AUC qui montre que le modèle apprend quelque chose.

## Solutions D

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
d = load_diabetes(as_frame=True)
pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(d.data, d.target)
print(pd.Series(pipe[-1].coef_, index=d.data.columns).abs().sort_values(ascending=False).head(2))   # s5 et s1 (s1 et s2 sont très colinéaires : leurs coefficients sont instables)

# 2
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
modeles = {
    "logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "arbre": DecisionTreeClassifier(max_depth=4, random_state=0),
    "forêt": RandomForestClassifier(random_state=0),
    "svm": make_pipeline(StandardScaler(), SVC()),
    "naive bayes": GaussianNB(),
}
print(pd.Series({n: m.fit(X_tr, y_tr).score(X_te, y_te) for n, m in modeles.items()}).sort_values(ascending=False).round(3))

# 3
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xi, yi, test_size=0.3, random_state=0, stratify=yi)
for k in [1, 5, 15, 30]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(Xa, ya)
    print(k, round(m.score(Xa, ya), 3), round(m.score(Xb, yb), 3))
# k petit : train≈1 (overfit potentiel) ; k très grand : les deux baissent (underfit)

# 4
from sklearn.datasets import make_moons
Xm, ym = make_moons(500, noise=0.25, random_state=0)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)
for nom, m in [("logistique", LogisticRegression()), ("arbre5", DecisionTreeClassifier(max_depth=5, random_state=0)), ("svm rbf", SVC())]:
    print(nom, round(m.fit(Xm_tr, ym_tr).score(Xm_te, ym_te), 3))
# la frontière de la régression logistique est une droite : elle ne sépare pas des demi-lunes entremêlées

# 5 : voir D.8 (silhouettes de K=2 et K=3 quasi identiques ; l’inertie montre le coude à K=3)

# 6
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
Xd = load_digits().data
p = PCA().fit(Xd)
print(int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= 0.90)) + 1)       # ≈ 21 composantes

# 7
def gini(lab):
    _, c = np.unique(lab, return_counts=True)
    return 1 - ((c / c.sum()) ** 2).sum()
print(gini(np.array([0, 0, 1, 1])))      # 0.5
```

## Solutions E

1. Total = 1000. Accuracy = (40+920)/1000 = **0,96** ; précision = 40/(40+30) = **0,571** ; rappel = 40/(40+10) = **0,8** ; F1 = 2·0,571·0,8/(1,371) ≈ **0,667** ; spécificité = 920/950 ≈ **0,968**. L'accuracy (96 %) masque le fait que 43 % des alertes positives sont fausses : la maladie est rare (5 %).
2. Rappel : maladie grave, sécurité (rater un cas est inacceptable). Précision : filtre anti-spam, recommandations coûteuses (fausses alertes inacceptables).
3. Par l'inégalité de Jensen/l'inégalité entre moyenne quadratique et moyenne arithmétique. Un RMSE bien supérieur à la MAE signale quelques erreurs très grandes.
4. Utiliser `validation_curve` comme en E.4 sur `load_wine` ; profondeur 3 suffit en général (~0,9+), l'illimitée ne fait pas mieux en validation (train = 1,0).
5. Utiliser `GridSearchCV(KNeighborsClassifier(), {"n_neighbors": range(1, 26), "weights": ["uniform", "distance"]}, cv=5)`. Le score test est souvent un peu différent du score CV ; ce n'est pas anormal : le test est un échantillon indépendant et fini, tandis que le score CV a été **sélectionné parmi plusieurs configurations** (léger optimisme).
6. Une faible profondeur donne un biais fort (modèle trop simple) et une variance faible ; une grande profondeur donne un biais faible mais une variance forte (l'arbre épouse le bruit).

## Solutions F

1. `HistGradientBoostingClassifier` est déjà dans `candidats` : comparez F1 et AUC. Ici l'écart avec la logistique est faible : la complexité supplémentaire ne se justifie pas.
2. `date_resiliation` n'existe que pour les clients qui ont résilié (NaN sinon) : le modèle apprend « date présente ⇒ résilié ». Score proche de 100 % en entraînement, inutilisable en production : **fuite de la cible**.
3. Avec une offre plus chère, on contacte moins de clients : le **seuil optimal monte** (on n'agit que sur les cas plus certains), car chaque FP coûte plus.
4. `def preparer(df): df = df.copy(); df.loc[df.facture_mensuelle < 0, "facture_mensuelle"] = np.nan; …` et on ajoute les trois features ; la même fonction est appelée à l'entraînement et à la prédiction.
5. Contenu minimal : problème, données, métrique, modèle retenu, résultats, commande de relance, limites.

---

# Conclusion

Vous disposez maintenant d'une vision complète : **comprendre** le problème, **préparer** les données sans fuite, **choisir** et **comprendre** les algorithmes, **évaluer** honnêtement, **construire** un projet de bout en bout, et **explorer** la suite. La compétence décisive n'est pas de connaître 50 algorithmes, mais de maîtriser la méthode : *baseline → validation rigoureuse → itération guidée par les erreurs → interprétation orientée décision*.
