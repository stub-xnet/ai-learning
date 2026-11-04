# Cours complet : NumPy et Pandas

> **Public visé** : débutant en analyse de données, avec des bases minimales de programmation.
> **Objectif** : maîtriser NumPy et Pandas jusqu'à pouvoir nettoyer, transformer et analyser des données réelles de façon autonome.
> **Prérequis technique** : Python 3.10+ ; NumPy ≥ 1.26 ; Pandas ≥ 2.0 (le cours a été vérifié avec NumPy 2.x et Pandas 2.3).

## Comment utiliser ce cours

1. Lisez un chapitre, **puis tapez vous-même les exemples** (ne copiez pas-collez : la frappe force la réflexion).
2. Modifiez les exemples : changez une valeur, cassez le code volontairement, lisez l'erreur.
3. Faites les questions et exercices de fin de chapitre **avant** de regarder les solutions (placées dans la section « Solutions des exercices du cours » à la fin).
4. Passez ensuite aux notebooks `01_DEBUTANT.ipynb`, `02_INTERMEDIAIRE.ipynb`, `03_AVANCE.ipynb`.

Convention des encadrés :

- 💡 **À retenir** : l'idée essentielle.
- ⚠️ **Erreur fréquente** : un piège classique, avec sa correction.
- 🎯 **Utile quand** : les situations réelles où la notion sert.

---

## Table des matières

- [Installation de l'environnement](#installation-de-lenvironnement)
- [PARTIE A — Rappels Python utiles](#partie-a--rappels-python-utiles)
  - A.1 Variables et types · A.2 Listes, tuples, dictionnaires · A.3 Conditions et boucles · A.4 Fonctions · A.5 Compréhensions · A.6 Indexation et slicing · A.7 Mutabilité · A.8 Imports et modules · Exercices A
- [PARTIE B — NumPy](#partie-b--numpy--maîtrise-complète)
  - B.0 Présentation · B.1 Installation · B.2 Création · B.3 1D/2D/nD · B.4 Indexation et slicing · B.5 Modification · B.6 Reshape/flatten · B.7 Transposition · B.8 Types · B.9–10 Génération · B.11 Opérations · B.12 Broadcasting · B.13–14 Statistiques · B.15 Axes · B.16–17 Masques et filtrage · B.18 Tri et recherche · B.19 Concaténation/séparation · B.20 Copie et vues · B.21 Aléatoire · B.22–23 Algèbre linéaire · B.24 Vectorisation · B.25 Erreurs et bonnes pratiques · Exercices B
- [PARTIE C — Pandas](#partie-c--pandas--maîtrise-complète)
  - C.1 à C.26 · NumPy + Pandas ensemble · Exercices C
- [PARTIE D — Erreurs classiques des débutants](#partie-d--erreurs-classiques-des-débutants)
- [Solutions des exercices du cours](#solutions-des-exercices-du-cours)
- [Aide-mémoire final](#aide-mémoire-final)

---

## Installation de l'environnement

Ne jamais installer de bibliothèques de data science dans le Python du système. On utilise un **environnement virtuel** (un dossier isolé contenant sa propre copie des bibliothèques).

```bash
cd ~/AI_LEARNING
python3 -m venv .venv                 # crée l'environnement (une seule fois)
source .venv/bin/activate             # l'active (à chaque nouvelle session de terminal)
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            jupyterlab ipykernel openpyxl pyarrow joblib xgboost
jupyter lab                           # ouvre l'interface pour les notebooks
```

Vérification :

```python
import numpy as np
import pandas as pd
print(np.__version__, pd.__version__)
```

> 💡 **À retenir** : `source .venv/bin/activate` doit être refait à chaque nouveau terminal. Le préfixe `(.venv)` apparaît alors dans l'invite.

---

# PARTIE A — Rappels Python utiles

Cette partie ne remplace pas un cours de Python. Elle ne couvre que ce dont vous avez **réellement** besoin pour NumPy et Pandas.

## A.1 Variables et types de données

Une **variable** est un nom qui référence une valeur en mémoire. Python est à **typage dynamique** : le type est porté par la valeur, pas par le nom.

```python
age = 21                 # int  : entier
prix = 19.99             # float: nombre à virgule flottante
nom = "Awa"              # str  : chaîne de caractères
est_etudiant = True      # bool : booléen
rien = None              # NoneType : « absence de valeur »

print(type(age), type(prix), type(nom), type(est_etudiant), type(rien))
# <class 'int'> <class 'float'> <class 'str'> <class 'bool'> <class 'NoneType'>
```

| Type | Exemple | Usage en data science |
|---|---|---|
| `int` | `42` | comptages, identifiants |
| `float` | `3.14` | mesures, prix, moyennes |
| `str` | `"Cotonou"` | catégories, noms, textes |
| `bool` | `True` | résultats de comparaisons, masques |
| `None` | `None` | valeur absente (≠ `0`, ≠ `""`) |

Conversions (« casting ») :

```python
int("12")        # 12
float("3.5")     # 3.5
str(100)         # '100'
int(7.9)         # 7  (tronque, n'arrondit PAS)
round(7.9)       # 8
```

⚠️ **Erreur fréquente** — `0.1 + 0.2 == 0.3` vaut `False`. Les flottants sont stockés en binaire, donc approximatifs. On compare avec `math.isclose` ou, plus tard, `np.isclose`.

```python
import math
print(0.1 + 0.2)                      # 0.30000000000000004
print(math.isclose(0.1 + 0.2, 0.3))   # True
```

## A.2 Listes, tuples et dictionnaires

### Liste (`list`) — séquence ordonnée et modifiable

```python
notes = [12, 15.5, 9, 18]
notes.append(14)          # ajoute à la fin
notes[0] = 13             # modifie le premier élément
print(len(notes), notes)  # 5 [13, 15.5, 9, 18, 14]
```

### Tuple (`tuple`) — séquence ordonnée **non modifiable**

```python
point = (3, 4)
x, y = point              # « dépaquetage »
# point[0] = 5            # TypeError : un tuple est immuable
```

Les tuples servent à représenter des valeurs qui ne doivent pas changer : coordonnées, **forme (`shape`) d'un tableau** NumPy `(3, 4)`.

### Dictionnaire (`dict`) — associations clé → valeur

```python
etudiant = {"nom": "Awa", "age": 21, "notes": [12, 15, 17]}
print(etudiant["nom"])               # Awa
etudiant["ville"] = "Cotonou"        # ajout d'une clé
print(etudiant.get("email", "n/a"))  # n/a (évite KeyError si la clé n'existe pas)
for cle, valeur in etudiant.items():
    print(cle, "->", valeur)
```

> 🎯 **Utile quand** : un `DataFrame` Pandas peut se construire à partir d'un dictionnaire `{nom_colonne: valeurs}` — c'est la manière la plus courante de créer de petits jeux de données.

| Structure | Ordonnée | Modifiable | Accès | Exemple d'usage |
|---|---|---|---|---|
| `list` | oui | oui | par position | suite de notes |
| `tuple` | oui | non | par position | shape, coordonnées |
| `dict` | oui (depuis 3.7) | oui | par clé | colonnes → valeurs |
| `set` | non | oui | appartenance | valeurs uniques |

## A.3 Conditions et boucles

```python
note = 14
if note >= 16:
    mention = "Très bien"
elif note >= 14:
    mention = "Bien"
elif note >= 10:
    mention = "Passable"
else:
    mention = "Ajourné"
print(mention)   # Bien
```

```python
total = 0
for n in [12, 15, 9]:
    total += n
print(total / 3)           # 12.0

for i, n in enumerate([12, 15, 9]):    # indice + valeur
    print(i, n)

for nom, n in zip(["Awa", "Koffi"], [15, 12]):   # parcours parallèle
    print(nom, n)

compteur = 0
while compteur < 3:
    compteur += 1
```

> 💡 **À retenir** : tout le but de NumPy et Pandas est de **remplacer ces boucles** par des opérations sur des tableaux entiers (« vectorisation », chapitre B.24). Plus rapide et plus lisible.

## A.4 Fonctions

```python
def moyenne(valeurs, arrondi=2):
    """Retourne la moyenne arithmétique d'une liste de nombres."""
    return round(sum(valeurs) / len(valeurs), arrondi)

print(moyenne([12, 15, 9]))               # 12.0
print(moyenne([12, 15, 10], arrondi=0))   # 12.0
```

- Les paramètres peuvent avoir des **valeurs par défaut**.
- On peut passer des arguments **par position** ou **par nom** (`arrondi=0`). Pandas et NumPy utilisent énormément les arguments nommés.
- Les fonctions `lambda` sont de petites fonctions anonymes :

```python
carre = lambda x: x ** 2
print(carre(5))   # 25
print(sorted(["Awa", "Koffi", "Ama"], key=lambda s: len(s)))   # ['Awa', 'Ama', 'Koffi']
```

> 🎯 **Utile quand** : `df["col"].apply(lambda x: ...)`, tri avec `key=`, etc.

## A.5 Compréhensions de listes

Une manière compacte de construire une liste.

```python
carres = [n ** 2 for n in range(5)]                    # [0, 1, 4, 9, 16]
pairs  = [n for n in range(10) if n % 2 == 0]          # [0, 2, 4, 6, 8]
statut = ["admis" if n >= 10 else "refusé" for n in [8, 12, 15]]
# ['refusé', 'admis', 'admis']
dico   = {n: n ** 2 for n in range(3)}                 # {0: 0, 1: 1, 2: 4}
```

Structure : `[expression for élément in itérable if condition]`.

## A.6 Indexation et slicing (sur les listes)

```python
v = [10, 20, 30, 40, 50]
#    0   1   2   3   4      indices positifs
#   -5  -4  -3  -2  -1      indices négatifs

v[0]      # 10
v[-1]     # 50  (dernier)
v[1:4]    # [20, 30, 40]   -> début INCLUS, fin EXCLUE
v[:2]     # [10, 20]
v[2:]     # [30, 40, 50]
v[::2]    # [10, 30, 50]   -> pas de 2
v[::-1]   # [50, 40, 30, 20, 10] -> inversé
```

Règle d'or : `v[début:fin:pas]` — la borne `fin` est **exclue**, donc `v[1:4]` contient `4 - 1 = 3` éléments. Cette convention est la même pour NumPy et pour `iloc` de Pandas (mais pas pour `loc` ! voir C.7).

## A.7 Mutabilité

Un objet **mutable** peut être modifié « en place » ; un objet **immuable** ne le peut pas.

| Mutables | Immuables |
|---|---|
| `list`, `dict`, `set`, **tableaux NumPy**, **DataFrames** | `int`, `float`, `str`, `tuple`, `bool` |

Conséquence cruciale : **l'affectation ne copie pas**, elle crée un second nom pour le même objet.

```python
a = [1, 2, 3]
b = a            # b désigne LE MÊME objet
b.append(4)
print(a)         # [1, 2, 3, 4] ← a a changé !

c = a.copy()     # vraie copie (superficielle)
c.append(5)
print(a, c)      # [1, 2, 3, 4] [1, 2, 3, 4, 5]
```

⚠️ **Erreur fréquente** : modifier un tableau NumPy ou un DataFrame en croyant travailler sur une copie. Nous y reviendrons en B.20 (copies et vues) et C (`SettingWithCopyWarning`).

## A.8 Imports et organisation des modules

```python
import math                       # le module complet : math.sqrt(4)
from math import sqrt, pi         # des noms précis : sqrt(4)
import numpy as np                # alias conventionnel
import pandas as pd               # alias conventionnel
from pathlib import Path          # manipulation de chemins
```

- Un **module** = un fichier `.py`. Un **paquet** = un dossier de modules.
- Les alias `np` et `pd` sont **universels** : utilisez-les toujours, tout le monde les comprendra.
- Évitez `from numpy import *` : il écrase des noms (`sum`, `min`, `max`, `abs`) et rend le code ambigu.
- Nommer un fichier `numpy.py` ou `pandas.py` casse l'import (Python importerait votre fichier) : classique !

```python
from pathlib import Path
dossier = Path.home() / "AI_LEARNING"
print(dossier.exists())
```

## Exercices A

1. Quel est le type de `3 / 2` ? de `3 // 2` ? de `3 % 2` ? Prédisez les valeurs puis vérifiez.
2. Écrivez une compréhension qui renvoie les carrés des nombres impairs de 1 à 15.
3. Que valent `v[1:-1]`, `v[::-2]` pour `v = [0, 1, 2, 3, 4, 5]` ?
4. Pourquoi `a = [1, 2]; b = a; b.append(3)` modifie-t-il `a` ?
5. Écrivez une fonction `mention(note)` qui renvoie « Bien » pour 14 ≤ note < 16, etc., puis appliquez-la à une liste par compréhension.

*(Solutions en fin de document.)*

---

# PARTIE B — NumPy : maîtrise complète

## B.0 Présentation

### Qu'est-ce que NumPy ?

**NumPy** (*Numerical Python*) est la bibliothèque fondamentale du calcul scientifique en Python. Son objet central est le **`ndarray`** (*N-dimensional array*) : un tableau de nombres de même type, stocké de façon compacte et contiguë en mémoire.

### Pourquoi l'utiliser ?

1. **Vitesse** : les boucles s'exécutent en C compilé, pas en Python interprété. Souvent 10 à 100 fois plus rapide.
2. **Concision** : `a + b` additionne deux tableaux entiers, sans boucle.
3. **Fondation de tout l'écosystème** : Pandas, Scikit-learn, SciPy, Matplotlib, TensorFlow, PyTorch reposent sur NumPy (ou sur son modèle de tableaux).

### Tableau NumPy vs liste Python

| Critère | Liste Python | `ndarray` NumPy |
|---|---|---|
| Types des éléments | mélangés possibles | **un seul type** (`dtype`) |
| Mémoire | pointeurs vers objets dispersés | bloc contigu compact |
| `a + b` | **concatène** | **additionne élément par élément** |
| Dimensions | listes imbriquées | véritable multi-dimension (`shape`) |
| Vitesse sur gros volumes | lente | très rapide |
| Taille | dynamique (`append`) | fixe à la création |

```python
import numpy as np

l1, l2 = [1, 2, 3], [10, 20, 30]
print(l1 + l2)                          # [1, 2, 3, 10, 20, 30]  (concaténation)
print(np.array(l1) + np.array(l2))      # [11 22 33]             (addition terme à terme)
```

Comparaison de vitesse :

```python
import time
n = 2_000_000
liste = list(range(n))
tab = np.arange(n)

t0 = time.perf_counter(); s1 = [x * 2 for x in liste]; t1 = time.perf_counter()
t2 = time.perf_counter(); s2 = tab * 2;                t3 = time.perf_counter()
print(f"liste : {t1 - t0:.3f}s | numpy : {t3 - t2:.4f}s")
# typiquement : liste : 0.08s | numpy : 0.003s (varie selon la machine)
```

### Vocabulaire fondamental

| Terme | Signification | Exemple pour `a = np.array([[1,2,3],[4,5,6]])` |
|---|---|---|
| **ndim** | nombre de dimensions (d'axes) | `2` |
| **shape** | tuple des tailles par axe | `(2, 3)` : 2 lignes, 3 colonnes |
| **size** | nombre total d'éléments | `6` |
| **dtype** | type des éléments | `int64` |
| **axis** | numéro d'une dimension | axe 0 = lignes (vertical), axe 1 = colonnes (horizontal) |
| **itemsize** | octets par élément | `8` |

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.ndim, a.shape, a.size, a.dtype, a.itemsize)
# 2 (2, 3) 6 int64 8
```

Visualisation mentale des axes pour un tableau 2D :

```
            axe 1 (colonnes) →
             col0 col1 col2
axe 0   lig0 [ 1    2    3 ]
(lignes) lig1 [ 4    5    6 ]
   ↓
```

> 💡 **À retenir** : « opérer **le long de l'axe k** » signifie que l'indice `k` varie et **disparaît** du résultat. `a.sum(axis=0)` somme sur les lignes → un résultat par colonne (shape `(3,)`). `a.sum(axis=1)` → un résultat par ligne (shape `(2,)`).

## B.1 Installation et importation

```bash
pip install numpy
```

```python
import numpy as np
print(np.__version__)
```

Pour afficher l'aide d'une fonction dans Jupyter : `np.mean?` ; hors Jupyter : `help(np.mean)`.

## B.2 Création des tableaux

### Depuis des listes

```python
a = np.array([1, 2, 3])                 # 1D
m = np.array([[1, 2], [3, 4]])          # 2D
f = np.array([1, 2.5, 3])               # float64 : promotion automatique
print(f, f.dtype)                       # [1.  2.5 3. ] float64
```

NumPy **choisit un type commun** : `int` + `float` → `float64` ; un `str` dans la liste → tout devient texte (à éviter).

```python
mix = np.array([1, "a", 3.5])
print(mix.dtype)    # <U32 : tout est converti en texte !
```

⚠️ **Erreur fréquente** : listes de longueurs inégales (« tableaux irréguliers »).

```python
# np.array([[1, 2], [3]])   → ValueError : forme inhomogène
```

Solution : rendre les lignes de même longueur, ou garder des listes/DataFrames.

### Fonctions de création (aperçu — détail en B.9/B.10)

| Fonction | Rôle |
|---|---|
| `np.zeros(shape)` | tableau de 0 |
| `np.ones(shape)` | tableau de 1 |
| `np.full(shape, v)` | rempli avec `v` |
| `np.eye(n)` | matrice identité |
| `np.arange(a, b, pas)` | suite avec pas |
| `np.linspace(a, b, n)` | `n` points équidistants |
| `np.empty(shape)` | non initialisé (contenu quelconque) |
| `np.zeros_like(x)` | même forme que `x` |

## B.3 Tableaux 1D, 2D et multidimensionnels

```python
v = np.array([5, 10, 15])                       # 1D : shape (3,)
M = np.array([[1, 2, 3], [4, 5, 6]])            # 2D : shape (2, 3)
T = np.arange(24).reshape(2, 3, 4)              # 3D : shape (2, 3, 4)
print(v.shape, M.shape, T.shape)                # (3,) (2, 3) (2, 3, 4)
```

Interprétation concrète d'un tableau 3D : `(2, 3, 4)` = 2 « blocs », chacun de 3 lignes et 4 colonnes. Exemple réel : 2 magasins × 3 produits × 4 trimestres de ventes.

```
T[0] = premier bloc (3x4)      T[1] = deuxième bloc (3x4)
```

Un tableau **couleur d'image** est typiquement `(hauteur, largeur, 3)` ; un lot d'images est 4D `(n_images, hauteur, largeur, 3)`.

⚠️ **Erreur fréquente** : confondre `(3,)` (vecteur 1D), `(1, 3)` (matrice-ligne) et `(3, 1)` (matrice-colonne). Ce sont trois objets différents qui se comportent différemment au broadcasting.

```python
v = np.array([1, 2, 3])
print(v.shape, v.reshape(1, 3).shape, v.reshape(3, 1).shape)
# (3,) (1, 3) (3, 1)
```

## B.4 Indexation et slicing

### 1D — identique aux listes

```python
a = np.array([10, 20, 30, 40, 50])
print(a[0], a[-1], a[1:4], a[::2])   # 10 50 [20 30 40] [10 30 50]
```

### 2D — `a[ligne, colonne]`

```python
M = np.array([[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12]])

print(M[1, 2])       # 7              un élément
print(M[0])          # [1 2 3 4]      une ligne entière
print(M[:, 1])       # [ 2  6 10]     une colonne entière
print(M[0:2, 1:3])   # [[2 3]
                     #  [6 7]]       sous-matrice
print(M[::2, ::2])   # [[ 1  3]
                     #  [ 9 11]]     une case sur deux
print(M[-1, -1])     # 12
```

Syntaxe : `M[sélection_axe0, sélection_axe1, ...]` ; `:` signifie « tout cet axe ».

### Indexation par liste (« fancy indexing »)

```python
print(M[[0, 2], :])        # lignes 0 et 2
print(M[:, [0, 3]])        # colonnes 0 et 3
print(M[[0, 1, 2], [0, 1, 2]])   # [ 1  6 11] : la diagonale (couples (0,0),(1,1),(2,2))
```

⚠️ **Différence capitale** : le *slicing* (`M[0:2]`) renvoie une **vue** (pas de copie) ; le *fancy indexing* (`M[[0,2]]`) renvoie une **copie**. Voir B.20.

### 3D

```python
T = np.arange(24).reshape(2, 3, 4)
print(T[1, 2, 3])      # 23 : bloc 1, ligne 2, colonne 3
print(T[:, 0, :].shape)   # (2, 4)
print(T[..., 0].shape)    # (2, 3) : « ... » = tous les axes restants
```

### Utile quand

Extraire une colonne d'un jeu de données (`M[:, 2]`), prélever les `n` premières observations (`M[:n]`), séparer variables et cible (`X = M[:, :-1]`, `y = M[:, -1]`) — gestes quotidiens en Machine Learning.

## B.5 Modification des tableaux

```python
a = np.array([1, 2, 3, 4, 5])
a[0] = 100            # un élément
a[1:3] = 0            # une tranche (diffusion de la valeur)
a[3:] = [7, 8]        # une tranche avec un tableau de même taille
print(a)              # [100   0   0   7   8]
```

```python
M = np.zeros((3, 3), dtype=int)
M[1, :] = 5           # toute la ligne 1
M[:, 2] = [1, 2, 3]   # toute la colonne 2
M[M == 0] = -1        # remplacement conditionnel (masque, B.16)
print(M)
# [[-1 -1  1]
#  [ 5  5  2]
#  [-1 -1  3]]
```

Ajouter/supprimer des éléments crée **un nouveau tableau** (taille fixe) :

```python
a = np.array([1, 2, 3])
b = np.append(a, 4)          # [1 2 3 4]
c = np.insert(a, 1, 99)      # [ 1 99  2  3]
d = np.delete(a, 0)          # [2 3]
```

⚠️ **Erreur fréquente** : insérer un float dans un tableau d'entiers : il est **tronqué silencieusement**.

```python
e = np.array([1, 2, 3])
e[0] = 9.99
print(e)       # [9 2 3] ← 9.99 devenu 9 sans aucune alerte !
```

Solution : créer le tableau avec `dtype=float` dès le départ.

## B.6 Reshape et flatten

`reshape` change la forme **sans changer les données** ; le produit des dimensions doit rester égal à `size`.

```python
a = np.arange(12)                 # [0 1 ... 11]
print(a.reshape(3, 4))            # 3 lignes, 4 colonnes
print(a.reshape(2, 6).shape)      # (2, 6)
print(a.reshape(2, -1).shape)     # (2, 6) : -1 = « calcule pour moi »
print(a.reshape(-1, 1).shape)     # (12, 1) : vecteur-colonne
```

L'ordre de remplissage par défaut est **ligne par ligne** (ordre C, « row-major »).

```python
M = np.arange(6).reshape(2, 3)
print(M.flatten())    # [0 1 2 3 4 5]  → COPIE aplatie
print(M.ravel())      # [0 1 2 3 4 5]  → vue si possible (plus économe)
print(M.reshape(-1))  # idem
```

| Méthode | Retourne | Remarque |
|---|---|---|
| `reshape` | vue (si possible) | forme libre |
| `ravel` | vue (si possible) | aplatit |
| `flatten` | **toujours copie** | aplatit |
| `np.newaxis` / `None` | vue | ajoute un axe de taille 1 |

```python
v = np.array([1, 2, 3])
print(v[:, np.newaxis].shape)   # (3, 1)
print(v[np.newaxis, :].shape)   # (1, 3)
print(np.squeeze(np.ones((1, 3, 1))).shape)   # (3,) : supprime les axes de taille 1
```

⚠️ **Erreur fréquente** : `a.reshape(5, 5)` sur 12 éléments → `ValueError: cannot reshape array of size 12 into shape (5,5)`. Vérifiez toujours `a.size`.

🎯 **Utile quand** : scikit-learn exige une matrice 2D `(n_observations, n_features)` ; un vecteur de caractéristique unique doit donc être transformé par `x.reshape(-1, 1)`.

## B.7 Transposition

La transposée échange les axes : l'élément `(i, j)` devient `(j, i)`.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(M.shape, M.T.shape)   # (2, 3) (3, 2)
```

Pour les tableaux d'ordre supérieur :

```python
T = np.zeros((2, 3, 4))
print(T.transpose(2, 0, 1).shape)    # (4, 2, 3) : permutation explicite des axes
print(np.swapaxes(T, 0, 2).shape)    # (4, 3, 2)
```

⚠️ `.T` sur un vecteur 1D ne fait **rien** : `np.array([1,2,3]).T.shape` vaut `(3,)`. Il faut passer par `reshape(-1, 1)`.

🎯 **Utile quand** : alignement de formes pour le produit matriciel, passage d'« observations en lignes » à « variables en lignes ».

## B.8 Types de données (dtype)

| dtype | Description | Plage / précision | Octets |
|---|---|---|---|
| `int8` / `uint8` | entier signé / non signé | −128..127 / 0..255 | 1 |
| `int16` | entier | ±32 767 | 2 |
| `int32` | entier | ±2,1·10⁹ | 4 |
| `int64` | entier (défaut sous Linux) | ±9,2·10¹⁸ | 8 |
| `float32` | décimal simple | ~7 chiffres | 4 |
| `float64` | décimal double (défaut) | ~15 chiffres | 8 |
| `bool` | vrai/faux | — | 1 |
| `<U10`, `str_` | texte | — | variable |
| `datetime64[ns]` | dates | — | 8 |

```python
a = np.array([1, 2, 3], dtype=np.float32)
b = a.astype(np.int64)         # conversion → renvoie une COPIE
print(a.dtype, b.dtype)        # float32 int64
print(np.array([1.9, -1.9]).astype(int))   # [ 1 -1] : troncature vers zéro
```

Débordement (*overflow*) silencieux :

```python
x = np.array([250], dtype=np.uint8)
print(x + 10)     # [4]  ← 260 mod 256 ! Aucune erreur.
```

⚠️ Les images en `uint8` débordent facilement : convertissez en `float32` avant de calculer.

Économie mémoire : un tableau de 1 million de `float64` pèse 8 Mo ; en `float32`, 4 Mo.

```python
big = np.zeros(1_000_000)
print(big.nbytes, big.astype(np.float32).nbytes)    # 8000000 4000000
```

### Valeurs spéciales : `nan` et `inf`

```python
x = np.array([1.0, np.nan, 3.0])
print(x.sum())              # nan : un NaN contamine tout calcul
print(np.nansum(x))         # 4.0
print(np.isnan(x))          # [False  True False]
print(np.nan == np.nan)     # False ! utilisez toujours np.isnan
print(1 / np.array([0.0]))  # [inf] (avec un avertissement)
```

`nan` n'existe que pour les `float` : un tableau d'entiers ne peut pas contenir de valeur manquante (Pandas contourne ce point, voir C.13).

## B.9 Génération de tableaux : les bases

```python
np.zeros((2, 3))                  # 2x3 de 0.
np.ones((2, 2), dtype=int)        # 2x2 de 1
np.full((2, 2), 7)                # 2x2 de 7
np.empty((2, 2))                  # contenu arbitraire (rapide, à écraser)
np.zeros_like(np.ones((3, 3)))    # même forme/type que l'argument
```

⚠️ `np.zeros(3, 3)` est une **erreur** : le deuxième argument est le `dtype`. La forme doit être un tuple : `np.zeros((3, 3))`.

## B.10 `arange`, `linspace`, `zeros`, `ones`, `full`, `eye`

### `arange(début, fin, pas)` — fin **exclue**

```python
print(np.arange(5))            # [0 1 2 3 4]
print(np.arange(2, 10, 3))     # [2 5 8]
print(np.arange(0, 1, 0.25))   # [0.   0.25 0.5  0.75]
```

⚠️ Avec des flottants, `arange` peut surprendre sur le nombre d'éléments (erreurs d'arrondi). Préférez `linspace`.

### `linspace(début, fin, n)` — fin **incluse**, `n` points

```python
print(np.linspace(0, 1, 5))    # [0.   0.25 0.5  0.75 1.  ]
print(np.linspace(0, 10, 3))   # [ 0.  5. 10.]
```

`np.logspace(0, 3, 4)` donne `[1, 10, 100, 1000]` (progression géométrique).

### `eye` et `identity`

```python
print(np.eye(3))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
print(np.diag([1, 2, 3]))    # matrice diagonale avec ces valeurs
```

| Fonction | Fin incluse ? | On fixe… |
|---|---|---|
| `arange` | non | le **pas** |
| `linspace` | oui | le **nombre de points** |

🎯 **Utile quand** : axes de graphiques (`linspace`), initialisation (`zeros`), matrice identité (`eye`) en algèbre linéaire, indices (`arange`).

## B.11 Opérations mathématiques

Toutes les opérations arithmétiques s'appliquent **élément par élément** (*element-wise*).

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)       # [11 22 33]
print(b - a)       # [ 9 18 27]
print(a * b)       # [10 40 90]      ← produit terme à terme, PAS matriciel
print(b / a)       # [10. 10. 10.]
print(a ** 2)      # [1 4 9]
print(b // 7)      # [1 2 4]
print(b % 7)       # [3 6 2]
print(a * 10)      # [10 20 30]      ← scalaire diffusé
```

Fonctions universelles (*ufuncs*) :

```python
x = np.array([0, np.pi / 2, np.pi])
print(np.sin(x).round(3))            # [0. 1. 0.]
print(np.sqrt(np.array([4, 9, 16])))   # [2. 3. 4.]
print(np.exp(1))                     # 2.718281828459045
print(np.log(np.e))                  # 1.0
print(np.log10(1000))                # 3.0
print(np.abs(np.array([-3, 2])))     # [3 2]
print(np.round(3.14159, 2))          # 3.14
print(np.floor(2.7), np.ceil(2.1))   # 2.0 3.0
print(np.clip(np.array([-5, 5, 15]), 0, 10))   # [ 0  5 10]
```

Comparaison élément par élément et vérification avec tolérance :

```python
print(np.isclose(0.1 + 0.2, 0.3))          # True
print(np.allclose([1, 2], [1.0000001, 2])) # True
```

⚠️ **Erreur fréquente** : opérer sur deux tableaux de formes incompatibles : `ValueError: operands could not be broadcast together with shapes (3,) (4,)`. Voir B.12.

⚠️ `a * b` n'est **pas** le produit matriciel : pour cela utilisez `a @ b` (B.23).

## B.12 Broadcasting

Le **broadcasting** (« diffusion ») est le mécanisme qui permet d'opérer sur des tableaux de formes différentes sans les dupliquer explicitement.

### Les règles

Pour comparer deux formes, NumPy aligne leurs dimensions **par la droite**. Deux dimensions sont compatibles si :
1. elles sont **égales**, ou
2. l'une vaut **1** (elle est alors étirée), ou
3. l'une est **absente** (considérée comme 1).

```
A      : (3, 4)
b      :    (4,)     → traité comme (1, 4)
résultat: (3, 4)     ✔

A      : (3, 4)
c      : (3, 1)
résultat: (3, 4)     ✔ (c étiré sur les colonnes)

A      : (3, 4)
d      : (3,)        → (1, 3) vs (3, 4) : 3 ≠ 4 ✘ ValueError
```

### Exemples

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M + 10)                      # scalaire diffusé partout
print(M + np.array([100, 200, 300]))   # ajoute à chaque LIGNE : shape (3,) → (1,3)
# [[101 202 303]
#  [104 205 306]]

print(M + np.array([[10], [20]]))  # shape (2,1) : ajoute à chaque COLONNE
# [[11 12 13]
#  [24 25 26]]
```

### Cas d'usage : centrer chaque colonne (soustraire la moyenne)

```python
rng = np.random.default_rng(0)
X = rng.normal(loc=[170, 65], scale=[10, 8], size=(5, 2))   # taille, poids
X_centre = X - X.mean(axis=0)        # (5,2) - (2,) → ok
print(X_centre.mean(axis=0).round(10))   # [0. 0.]
```

### Piège : normaliser chaque **ligne**

```python
M = np.array([[1., 2., 3.],
              [4., 5., 6.]])
somme_lignes = M.sum(axis=1)             # shape (2,)
# M / somme_lignes                        # ✘ ValueError (3 vs 2)
print(M / somme_lignes[:, np.newaxis])   # ✔ (2,1) : chaque ligne somme à 1
# ou : M / M.sum(axis=1, keepdims=True)
```

> 💡 **À retenir** : `keepdims=True` conserve l'axe réduit avec une taille 1, ce qui rend le broadcasting immédiat. Réflexe à prendre.

### Produit « extérieur » par broadcasting : table de multiplication

```python
a = np.arange(1, 4).reshape(-1, 1)    # (3,1)
b = np.arange(1, 5)                   # (4,)
print(a * b)                          # (3,4)
# [[ 1  2  3  4]
#  [ 2  4  6  8]
#  [ 3  6  9 12]]
```

### Distances entre tous les couples de points (sans boucle)

```python
P = np.array([[0, 0], [3, 4], [6, 8]])
diff = P[:, None, :] - P[None, :, :]       # (3,1,2) - (1,3,2) → (3,3,2)
D = np.sqrt((diff ** 2).sum(axis=-1))      # (3,3)
print(D)
# [[ 0.  5. 10.]
#  [ 5.  0.  5.]
#  [10.  5.  0.]]
```

C'est exactement le calcul de base des algorithmes K-NN et K-Means (voir cours Machine Learning).

## B.13 Fonctions statistiques

```python
notes = np.array([12, 15, 9, 18, 14, 11, 16])

print(notes.sum())            # 95
print(notes.mean())           # 13.571428571428571
print(np.median(notes))       # 14.0
print(notes.min(), notes.max())    # 9 18
print(np.ptp(notes))          # 9    (max - min, « peak to peak »)
print(notes.std())            # écart-type de la population (ddof=0)
print(notes.std(ddof=1))      # écart-type de l'échantillon (ddof=1)
print(notes.var(ddof=1))      # variance échantillon
print(np.percentile(notes, [25, 50, 75]))   # quartiles
print(notes.argmax(), notes.argmin())       # 3 2 : POSITIONS du max / min
print(notes.cumsum())         # [12 27 36 54 68 79 95]
```

| Statistique | Fonction | Interprétation |
|---|---|---|
| Moyenne | `mean` | centre « arithmétique », sensible aux valeurs extrêmes |
| Médiane | `median` | valeur qui coupe l'échantillon en deux, robuste |
| Écart-type | `std` | dispersion typique autour de la moyenne |
| Variance | `var` | écart-type² |
| Percentiles | `percentile` | seuils (25 % des valeurs sont sous le Q1) |
| Corrélation | `corrcoef` | liaison linéaire entre -1 et 1 |

⚠️ **`ddof`** : NumPy calcule par défaut l'écart-type de la **population** (division par `n`) ; Pandas calcule celui de l'**échantillon** (division par `n-1`). Même données, résultats différents :

```python
x = np.array([2., 4., 4., 4., 5., 5., 7., 9.])
print(np.std(x))                 # 2.0
print(np.std(x, ddof=1))         # 2.138...
print(pd.Series(x).std())        # 2.138...  (Pandas : ddof=1 par défaut)
```

Valeurs extrêmes : moyenne vs médiane.

```python
salaires = np.array([300, 320, 310, 305, 5000])
print(salaires.mean(), np.median(salaires))    # 1247.0 310.0
```

Une seule valeur aberrante (« outlier ») fait exploser la moyenne : la médiane est plus fidèle.

## B.14 Sommes, moyennes, médianes et écarts-types par axe

```python
ventes = np.array([[120, 135, 150],     # magasin A (jan, fév, mar)
                   [ 90,  95, 110],     # magasin B
                   [200, 180, 220]])    # magasin C

print(ventes.sum())            # 1300 : total général
print(ventes.sum(axis=0))      # [410 410 480] : total par MOIS (on écrase les magasins)
print(ventes.sum(axis=1))      # [405 295 600] : total par MAGASIN
print(ventes.mean(axis=1).round(1))   # [135.   98.3 200. ]
print(np.median(ventes, axis=0))      # [120. 135. 150.]
print(ventes.std(axis=0).round(2))
print(ventes.max(axis=1))             # [150 110 220]
print(ventes.argmax(axis=1))          # [2 2 2] : mois du maximum par magasin
```

Moyennes en ignorant les `nan` :

```python
d = np.array([[1., np.nan], [3., 4.]])
print(np.nanmean(d, axis=0))       # [2. 4.]
```

**Règle mnémotechnique** : `axis=0` → on « écrase » les lignes (résultat : une valeur par colonne) ; `axis=1` → on « écrase » les colonnes (une valeur par ligne). Dans le doute : regardez la `shape` du résultat.

## B.15 Manipulation des axes

```python
A = np.arange(24).reshape(2, 3, 4)
print(A.sum(axis=0).shape)          # (3, 4)
print(A.sum(axis=(0, 2)).shape)     # (3,)  : plusieurs axes à la fois
print(A.sum(axis=-1).shape)         # (2, 3): dernier axe
print(np.moveaxis(A, 0, -1).shape)  # (3, 4, 2)
print(A.sum(axis=1, keepdims=True).shape)   # (2, 1, 4)
```

Fonctions utiles : `np.apply_along_axis(f, axis, arr)` applique une fonction le long d'un axe (lent : préférez les opérations vectorisées), `np.expand_dims(a, axis)` ajoute un axe.

## B.16 Comparaisons et masques booléens

Une comparaison produit un tableau de booléens, appelé **masque**.

```python
notes = np.array([12, 15, 9, 18, 14, 11, 16])
masque = notes >= 14
print(masque)             # [False  True False  True  True False  True]
print(masque.sum())       # 4  (True compte pour 1)
print(masque.mean())      # 0.571... : proportion de notes ≥ 14
```

Combiner des conditions : `&` (et), `|` (ou), `~` (non), **avec parenthèses** :

```python
print((notes >= 10) & (notes < 15))    # entre 10 et 14 inclus
print((notes < 10) | (notes > 17))
print(~(notes >= 14))
```

⚠️ **Erreur fréquente** :
- `notes >= 10 and notes < 15` → `ValueError: The truth value of an array ... is ambiguous`. Utilisez `&`.
- `notes >= 10 & notes < 15` (sans parenthèses) échoue aussi : `&` est prioritaire sur les comparaisons.

Tests globaux :

```python
print(np.any(notes > 17), np.all(notes >= 9))   # True True
print(np.isin(notes, [9, 18]))                  # test d'appartenance
```

`np.where` :

```python
print(np.where(notes >= 10, "admis", "refusé"))
print(np.where(notes >= 14)[0])      # [1 3 4 6] : INDICES des éléments vrais
```

Choix multiples : `np.select`.

```python
conds = [notes >= 16, notes >= 14, notes >= 10]
choix = ["TB", "B", "P"]
print(np.select(conds, choix, default="AJ"))
# ['P' 'B' 'AJ' 'TB' 'B' 'P' 'TB']
```

## B.17 Filtrage

Indexer par un masque conserve les éléments `True`.

```python
print(notes[notes >= 14])        # [15 18 14 16]
print(notes[(notes > 10) & (notes < 16)])   # [12 15 14 11]
```

Sur un tableau 2D :

```python
data = np.array([[25, 1200], [31, 2100], [45, 3000], [52, 2800]])   # âge, salaire
ages, sal = data[:, 0], data[:, 1]
print(data[ages > 30])                  # lignes dont l'âge > 30
print(sal[ages > 30].mean())            # salaire moyen des plus de 30 ans : 2633.33...
data[sal > 2500, 1] = 2500              # plafonner les salaires (modification par masque)
```

Le filtrage renvoie toujours une **copie** (le tableau filtré est indépendant).

Éliminer les valeurs manquantes :

```python
x = np.array([1., np.nan, 3., np.nan, 5.])
print(x[~np.isnan(x)])     # [1. 3. 5.]
```

## B.18 Tri et recherche

```python
a = np.array([30, 10, 20, 40])
print(np.sort(a))           # [10 20 30 40]    copie triée
print(np.sort(a)[::-1])     # [40 30 20 10]    décroissant
print(np.argsort(a))        # [1 2 0 3]        indices qui trieraient a
a.sort()                    # tri EN PLACE (modifie a)
```

`argsort` permet de trier un tableau selon un autre :

```python
noms = np.array(["Awa", "Koffi", "Ama", "Yao"])
notes = np.array([15, 12, 18, 9])
ordre = np.argsort(-notes)          # décroissant
print(noms[ordre])                  # ['Ama' 'Awa' 'Koffi' 'Yao']
print(noms[np.argsort(notes)[-2:]]) # les 2 meilleurs
```

Tri 2D :

```python
M = np.array([[3, 1], [1, 2], [2, 0]])
print(np.sort(M, axis=0))             # chaque colonne triée indépendamment
print(M[np.argsort(M[:, 0])])         # lignes triées selon la colonne 0 (lignes conservées)
```

Recherche :

```python
s = np.array([10, 20, 30, 40])
print(np.searchsorted(s, 25))     # 2 : position d'insertion (tableau trié requis)
print(np.unique(np.array([3, 1, 3, 2, 1])))   # [1 2 3]
vals, comptes = np.unique(np.array(["a", "b", "a", "a"]), return_counts=True)
print(dict(zip(vals, comptes)))   # {'a': 3, 'b': 1}
print(np.argmax(s), np.nonzero(s > 15)[0])    # 3 [1 2 3]
```

⚠️ **Erreur fréquente** : `np.sort(M)` sans `axis` trie **chaque ligne** (axe -1 par défaut), ce qui détruit la correspondance entre colonnes. Pour trier des *lignes* selon une colonne, utilisez `argsort`.

## B.19 Concaténation et séparation

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(np.concatenate([a, b], axis=0))   # empile verticalement  (4,2)
print(np.concatenate([a, b], axis=1))   # juxtapose             (2,4)
print(np.vstack([a, b]).shape, np.hstack([a, b]).shape)   # (4, 2) (2, 4)
print(np.stack([a, b]).shape)           # (2, 2, 2) : NOUVEL axe
print(np.stack([a, b], axis=-1).shape)  # (2, 2, 2)
```

| Fonction | Effet | Axe nouveau ? |
|---|---|---|
| `concatenate(axis=0)` / `vstack` | empile en hauteur | non |
| `concatenate(axis=1)` / `hstack` | juxtapose | non |
| `stack` | empile en créant un axe | **oui** |
| `column_stack` | vecteurs 1D → colonnes | — |

```python
x = np.array([1, 2, 3]); y = np.array([4, 5, 6])
print(np.column_stack([x, y]))     # [[1 4] [2 5] [3 6]]
```

Séparation :

```python
v = np.arange(10)
print(np.split(v, 5))                 # 5 morceaux égaux
print(np.array_split(v, 3))           # 3 morceaux (tailles 4,3,3), tolère l'inégalité
print(np.split(v, [3, 7]))            # coupe aux indices 3 et 7 → [0..2], [3..6], [7..9]
M = np.arange(12).reshape(3, 4)
haut, bas = np.vsplit(M, [2]); gauche, droite = np.hsplit(M, 2)
```

⚠️ `np.concatenate` exige que les formes concordent sur **tous** les axes **sauf** l'axe concerné. `np.concatenate([a, np.array([1,2,3])])` échoue (dimensions différentes) : utilisez `[[…]]` ou `reshape`.

🎯 **Utile quand** : ajouter une colonne de 1 (biais) à une matrice `X` : `np.hstack([np.ones((len(X), 1)), X])` ; regrouper des mesures en lot.

## B.20 Copie et vues

Une **vue** partage les mêmes données en mémoire qu'un autre tableau ; une **copie** est indépendante.

```python
a = np.arange(6)
v = a[1:4]          # slicing → VUE
v[0] = 99
print(a)            # [ 0 99  2  3  4  5] ← a modifié !

c = a[1:4].copy()   # copie explicite
c[0] = -1
print(a)            # inchangé par c
```

| Opération | Résultat |
|---|---|
| Slicing `a[1:4]` | **vue** |
| `reshape`, `ravel`, `.T`, `transpose` | **vue** (en général) |
| Fancy indexing `a[[1,2]]` | **copie** |
| Masque booléen `a[a>2]` | **copie** |
| `.copy()`, `flatten()`, `astype()` | **copie** |
| `b = a` | **même objet** (ni vue ni copie) |

Vérifier si deux tableaux partagent la mémoire :

```python
print(np.shares_memory(a, a[1:4]), np.shares_memory(a, a[[1, 2]]))   # True False
print(a[1:4].base is a)    # True
```

⚠️ **Erreur fréquente** : modifier une sous-partie en pensant isoler le tableau d'origine. Règle : **dès que vous voulez modifier une extraction sans toucher l'original, faites `.copy()`**.

> 💡 Les vues sont une force (économie de mémoire, rapidité) et un danger (effets de bord). Soyez-en conscient à chaque slicing suivi d'une modification.

## B.21 Génération aléatoire

L'interface moderne est le **générateur** `default_rng` (préférez-la à `np.random.seed` / `np.random.rand`).

```python
rng = np.random.default_rng(42)       # 42 = graine → résultats reproductibles
print(rng.random(3))                  # 3 flottants dans [0, 1)
print(rng.integers(1, 7, size=5))     # 5 lancers de dé (1..6, 7 exclu)
print(rng.normal(loc=10, scale=2, size=3))      # loi normale (moyenne 10, écart-type 2)
print(rng.uniform(5, 10, size=2))
print(rng.choice(["rouge", "vert", "bleu"], size=4, p=[0.5, 0.3, 0.2]))
print(rng.permutation(5))             # permutation aléatoire de 0..4
print(rng.choice(10, size=3, replace=False))   # tirage sans remise
```

La **graine** (*seed*) rend l'expérience reproductible : mêmes graines → mêmes nombres. Indispensable pour comparer des modèles de Machine Learning.

Simulation : estimer π par Monte-Carlo.

```python
rng = np.random.default_rng(0)
pts = rng.random((1_000_000, 2))
dans_cercle = (pts ** 2).sum(axis=1) <= 1
print(4 * dans_cercle.mean())      # ≈ 3.14
```

Dataset synthétique réaliste :

```python
rng = np.random.default_rng(1)
n = 200
surface = rng.uniform(20, 150, n)                      # m²
prix = 800 * surface + rng.normal(0, 15000, n)         # relation linéaire + bruit
print(np.corrcoef(surface, prix)[0, 1].round(2))       # proche de 1
```

⚠️ Ne pas confondre `integers(a, b)` (b exclu) et `uniform`. Et ne jamais réutiliser la même graine pour des tirages censés être indépendants.

## B.22 Algèbre linéaire

Le sous-module `np.linalg` couvre les opérations de base.

```python
A = np.array([[2., 1.], [1., 3.]])
b = np.array([5., 10.])

print(np.linalg.det(A))              # 5.0 (déterminant)
print(np.linalg.inv(A))              # inverse
x = np.linalg.solve(A, b)            # résout A x = b
print(x)                             # [1. 3.]
print(np.allclose(A @ x, b))         # True : vérification

valeurs, vecteurs = np.linalg.eig(A) # valeurs/vecteurs propres
print(valeurs.round(3))
print(np.linalg.norm(np.array([3., 4.])))   # 5.0 (norme euclidienne)
print(np.linalg.matrix_rank(A))      # 2
```

⚠️ Préférez **`solve`** à `inv(A) @ b` : plus rapide et numériquement plus stable. Une matrice singulière (déterminant nul) lève `LinAlgError: Singular matrix`.

### Moindres carrés (régression linéaire)

```python
x = np.array([1., 2., 3., 4., 5.])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
X = np.column_stack([np.ones_like(x), x])         # colonne de 1 + variable
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(beta.round(3))      # [0.05 1.99] → y ≈ 0.05 + 1.99·x
```

C'est la méthode derrière la régression linéaire (voir cours Machine Learning).

## B.23 Opérations matricielles

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])

print(A * B)            # terme à terme (Hadamard)
print(A @ B)            # produit matriciel : [[2 1] [4 3]]
print(np.dot(A, B))     # équivalent pour 2D
print(A.T)              # transposée
print(np.trace(A))      # 5 : somme de la diagonale
print(np.diag(A))       # [1 4]
```

**Règle du produit matriciel** : `(m, n) @ (n, p) → (m, p)` — les dimensions « intérieures » doivent être égales.

```python
X = np.ones((100, 3)); w = np.array([0.5, -1.0, 2.0])
print((X @ w).shape)   # (100,) : une prédiction par observation
```

Produit scalaire, produit vectoriel, produit externe :

```python
u = np.array([1, 2, 3]); v = np.array([4, 5, 6])
print(u @ v)               # 32 = 1·4 + 2·5 + 3·6
print(np.cross(u, v))      # [-3  6 -3]
print(np.outer(u, v).shape)   # (3, 3)
```

`np.einsum` : notation compacte pour des contractions complexes.

```python
print(np.einsum("ij,jk->ik", A, B))      # équivaut à A @ B
```

⚠️ `A * B` ≠ `A @ B`. C'est LA confusion n°1 en algèbre linéaire avec NumPy.

🎯 **Utile quand** : prédictions d'un modèle linéaire (`X @ w`), couches d'un réseau de neurones, ACP, covariance.

## B.24 Optimisation et vectorisation

**Vectoriser**, c'est remplacer une boucle Python par une opération sur tableau entier.

```python
rng = np.random.default_rng(0)
x = rng.random(1_000_000)

# Version boucle
def somme_carres_boucle(x):
    s = 0.0
    for v in x:
        s += v * v
    return s

# Version vectorisée
def somme_carres_vec(x):
    return (x * x).sum()      # ou np.dot(x, x)

import time
t = time.perf_counter(); r1 = somme_carres_boucle(x); t1 = time.perf_counter() - t
t = time.perf_counter(); r2 = somme_carres_vec(x);    t2 = time.perf_counter() - t
print(np.isclose(r1, r2), f"boucle {t1:.3f}s vs vectorisé {t2:.5f}s")
```

Autres techniques :

| Technique | Exemple |
|---|---|
| Éviter les boucles | `a * 2` plutôt que `[x*2 for x in a]` |
| Masques au lieu de `if` | `np.where(a > 0, a, 0)` au lieu d'une boucle conditionnelle |
| Pré-allouer | `np.empty(n)` plutôt que `append` en boucle |
| `out=` | `np.add(a, b, out=a)` évite un tableau temporaire |
| Bon `dtype` | `float32` divise la mémoire par 2 |
| Opérations par axe | `A.sum(axis=1)` plutôt qu'une boucle sur les lignes |
| `np.vectorize` | ⚠️ simple confort syntaxique, **pas** plus rapide |

Exemple complet : moyenne mobile sans boucle.

```python
x = np.arange(10, dtype=float)
k = 3
fenetre = np.ones(k) / k
print(np.convolve(x, fenetre, mode="valid"))    # [1. 2. 3. 4. 5. 6. 7. 8.]
```

Mesurer avant d'optimiser (dans Jupyter) : `%timeit tab.sum()`. « Premature optimization is the root of all evil » : écrivez d'abord un code correct, mesurez, puis optimisez.

## B.25 Erreurs fréquentes et bonnes pratiques

| Symptôme | Cause | Remède |
|---|---|---|
| `ValueError: operands could not be broadcast` | formes incompatibles | afficher `.shape`, `reshape`, `[:, None]` |
| Résultat « entier » inattendu | dtype `int` | `dtype=float`, `.astype(float)` |
| Modification inopinée de l'original | vue | `.copy()` |
| `truth value of an array is ambiguous` | `and`/`or`/`if arr` | `&`, `|`, `.any()`, `.all()` |
| `nan` partout | valeurs manquantes | `np.nanmean`, filtrer `~np.isnan` |
| Mauvais axe | confusion lignes/colonnes | tester sur un petit exemple, lire `shape` |
| `(n,)` vs `(n,1)` | vecteur 1D vs colonne | `reshape(-1, 1)` |
| Troncature silencieuse | affecter un float à un int | tableau en `float` |
| Débordement `uint8` | plage dépassée | `astype(np.float32)` |
| `A * B` pour un produit matriciel | confusion | `A @ B` |

Bonnes pratiques :

1. **Toujours** afficher `shape` et `dtype` à chaque étape d'un calcul inconnu.
2. Tester sur un petit exemple (3×4) avant les gros volumes.
3. Utiliser `default_rng(graine)` pour la reproductibilité.
4. Nommer clairement : `X` (matrice de variables), `y` (cible), `n_obs`, `n_feat`.
5. Préférer les fonctions vectorisées aux boucles.
6. Copier explicitement quand on modifie une extraction.
7. Documenter la forme attendue des tableaux dans les docstrings : `X : ndarray de shape (n, p)`.

## Exemples concrets d'analyse de données avec NumPy

**Cas 1 — Notes d'étudiants** : normalisation « z-score » par matière.

```python
rng = np.random.default_rng(7)
notes = rng.normal(loc=[12, 14, 10], scale=[3, 2, 4], size=(30, 3)).clip(0, 20).round(1)
# 30 étudiants × 3 matières (maths, français, physique)

moyenne_matiere = notes.mean(axis=0)
ecart_matiere = notes.std(axis=0)
z = (notes - moyenne_matiere) / ecart_matiere
print(z.mean(axis=0).round(8), z.std(axis=0).round(8))   # [0. 0. 0.] [1. 1. 1.]

moyenne_etudiant = notes.mean(axis=1)
meilleurs = np.argsort(moyenne_etudiant)[::-1][:3]
print("Top 3 (indices) :", meilleurs, moyenne_etudiant[meilleurs].round(2))
print("Admis (moy ≥ 10) :", (moyenne_etudiant >= 10).sum(), "/", len(notes))
```

**Cas 2 — Détection de valeurs aberrantes (règle de l'IQR)**

```python
dep = np.array([32, 28, 35, 30, 29, 31, 33, 27, 410, 30])      # dépenses journalières
q1, q3 = np.percentile(dep, [25, 75])
iqr = q3 - q1
bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(dep[(dep < bas) | (dep > haut)])     # [410]
```

**Cas 3 — Évolution des ventes : variation en pourcentage**

```python
ca = np.array([100., 110., 99., 120.])
var = (ca[1:] - ca[:-1]) / ca[:-1] * 100
print(var.round(1))    # [ 10.  -10.   21.2]
```

**Cas 4 — Corrélation entre variables**

```python
heures = np.array([1, 2, 3, 4, 5, 6, 7, 8])
note = np.array([8, 9, 11, 12, 13, 15, 16, 18])
print(np.corrcoef(heures, note).round(3))
```

## Exercices B

**Questions de compréhension**

1. Quelle est la différence entre `shape`, `size` et `ndim` ? Donnez-les pour `np.zeros((4, 5, 2))`.
2. Pourquoi `np.array([1, 2, 3]) + np.array([1, 2])` échoue-t-il, alors que `np.array([[1],[2],[3]]) + np.array([1, 2])` fonctionne ?
3. Quelle est la différence entre une vue et une copie ? Citez deux opérations qui produisent chacune.
4. Pourquoi `np.nan == np.nan` vaut-il `False` ?
5. Quand utiliser `arange` et quand `linspace` ?

**Petits exercices**

6. Créez le tableau 1D des multiples de 5 de 0 à 100, puis sélectionnez ceux qui sont des multiples de 3.
7. Créez une matrice 5×5 avec des 1 sur la bordure et des 0 à l'intérieur.
8. Soit `M = np.arange(20).reshape(4, 5)`. Extrayez : la dernière colonne ; les lignes 1 et 3 ; la sous-matrice des 2 premières lignes / 3 dernières colonnes.
9. Calculez, pour une matrice de ventes (6 produits × 12 mois) générée aléatoirement, le meilleur mois de chaque produit et le produit le plus vendu sur l'année.
10. Normalisez min-max (valeurs entre 0 et 1) chaque colonne d'une matrice 100×4.
11. Écrivez une fonction qui renvoie, pour un vecteur de notes, la proportion de notes au-dessus de la moyenne.
12. Calculez la matrice de distances euclidiennes entre 5 points aléatoires de ℝ³ sans boucle.

*(Solutions en fin de document.)*

---
# PARTIE C — Pandas : maîtrise complète

## C.1 Présentation de Pandas

**Pandas** est la bibliothèque de référence pour manipuler des **données tabulaires** (lignes × colonnes avec étiquettes), à la manière d'un tableur Excel ou d'une table SQL, mais programmable et capable de traiter des millions de lignes.

| Besoin | NumPy | Pandas |
|---|---|---|
| Calcul numérique rapide | ✔✔ | ✔ (s'appuie sur NumPy) |
| Colonnes de types différents | ✘ | ✔ |
| Étiquettes de lignes/colonnes | ✘ | ✔ |
| Valeurs manquantes | difficile | ✔ natif |
| Lecture CSV/Excel/SQL/JSON | ✘ | ✔ |
| Regroupements, jointures, dates | ✘ | ✔ |

Deux structures :

- **Series** : une colonne (1D) étiquetée par un **index**.
- **DataFrame** : une table (2D) = un dictionnaire de Series partageant le même index.

```python
import numpy as np
import pandas as pd
print(pd.__version__)
```

## C.2 Series et DataFrame

### Series

```python
s = pd.Series([12, 15, 9], index=["Awa", "Koffi", "Ama"], name="note")
print(s)
# Awa      12
# Koffi    15
# Ama       9
# Name: note, dtype: int64

print(s["Koffi"])        # 15      accès par étiquette
print(s.iloc[0])         # 12      accès par position
print(s.values)          # [12 15  9]  → tableau NumPy sous-jacent
print(s.index.tolist())  # ['Awa', 'Koffi', 'Ama']
print(s * 2)             # opérations vectorisées, index conservé
```

Alignement automatique sur l'index (grande différence avec NumPy) :

```python
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["y", "z", "w"])
print(a + b)
# w     NaN
# x     NaN
# y    12.0
# z    23.0
# dtype: float64
```

### DataFrame

```python
df = pd.DataFrame({
    "nom":   ["Awa", "Koffi", "Ama", "Yao"],
    "age":   [21, 23, 22, 24],
    "note":  [15.5, 12.0, 18.0, 9.5],
    "ville": ["Cotonou", "Porto-Novo", "Cotonou", "Parakou"],
})
print(df)
#      nom  age  note       ville
# 0    Awa   21  15.5     Cotonou
# 1  Koffi   23  12.0  Porto-Novo
# 2    Ama   22  18.0     Cotonou
# 3    Yao   24   9.5     Parakou
```

Anatomie d'un DataFrame :

```
            colonnes (df.columns)
            nom   age   note   ville
index  0    ...
(df.index) 1
           2
```

Attributs essentiels : `df.shape`, `df.columns`, `df.index`, `df.dtypes`, `df.values` (ou `df.to_numpy()`).

## C.3 Création de données

```python
# 1. dictionnaire de colonnes (le plus courant)
df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

# 2. liste de dictionnaires (un par ligne, typique des API JSON)
df2 = pd.DataFrame([{"a": 1, "b": 3}, {"a": 2, "b": 4}])

# 3. tableau NumPy + noms de colonnes
df3 = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])

# 4. liste de listes
df4 = pd.DataFrame([[1, "a"], [2, "b"]], columns=["num", "lettre"])

# 5. index personnalisé
df5 = pd.DataFrame({"prix": [100, 200]}, index=["pomme", "poire"])
```

Séries de dates :

```python
dates = pd.date_range("2025-01-01", periods=5, freq="D")
print(dates)
```

Dataset synthétique réaliste de ventes (utilisé tout au long du cours) :

```python
rng = np.random.default_rng(42)
n = 500
ventes = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=n, freq="D").to_series().sample(n, random_state=1).values,
    "produit": rng.choice(["Riz", "Huile", "Savon", "Sucre", "Lait"], n),
    "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], n, p=[0.2, 0.4, 0.2, 0.2]),
    "quantite": rng.integers(1, 20, n),
    "prix_unitaire": rng.choice([650, 1200, 400, 800, 950], n),
})
ventes["chiffre_affaires"] = ventes["quantite"] * ventes["prix_unitaire"]
print(ventes.head())
```

⚠️ Dans un jeu de données cohérent, le prix dépend du produit. Plus loin, nous construirons des datasets où cette cohérence est respectée.

## C.4 Importation CSV, Excel et JSON

### Créer un fichier d'exemple

```python
from pathlib import Path
Path("data").mkdir(exist_ok=True)
df.to_csv("data/etudiants.csv", index=False)
```

### CSV

```python
df = pd.read_csv("data/etudiants.csv")
```

Paramètres importants de `read_csv` :

| Paramètre | Rôle | Exemple |
|---|---|---|
| `sep` | séparateur | `sep=";"` (CSV français) |
| `decimal` | séparateur décimal | `decimal=","` |
| `encoding` | encodage | `encoding="utf-8"` ou `"latin-1"` |
| `header` | ligne des noms | `header=None` si absente |
| `names` | noms imposés | `names=["a","b"]` |
| `usecols` | colonnes à lire | `usecols=["nom","note"]` |
| `index_col` | colonne d'index | `index_col="id"` |
| `parse_dates` | colonnes de dates | `parse_dates=["date"]` |
| `dtype` | types imposés | `dtype={"code": str}` |
| `na_values` | marqueurs de NA | `na_values=["?", "N/A", ""]` |
| `nrows` | nombre de lignes | `nrows=1000` |
| `skiprows` | lignes à sauter | `skiprows=2` |
| `chunksize` | lecture par blocs | `chunksize=100_000` |

Fichier français typique (Excel exporte souvent avec `;` et `,`) :

```python
from io import StringIO
texte = "nom;note\nAwa;15,5\nKoffi;12,0"
print(pd.read_csv(StringIO(texte), sep=";", decimal=","))
```

⚠️ **Erreurs fréquentes** : tout dans une seule colonne (mauvais `sep`) ; caractères étranges `Ã©` (mauvais `encoding`) ; codes postaux `01200` devenus `1200` (forcer `dtype=str`).

### Excel (nécessite `openpyxl`)

```python
df.to_excel("data/etudiants.xlsx", index=False, sheet_name="Notes")
df_x = pd.read_excel("data/etudiants.xlsx", sheet_name="Notes")
```

### JSON

```python
df.to_json("data/etudiants.json", orient="records", force_ascii=False, indent=2)
df_j = pd.read_json("data/etudiants.json")
```

Pour les JSON imbriqués : `pd.json_normalize(liste_de_dicts)`.

### Autres sources

`pd.read_sql(requete, connexion)`, `pd.read_parquet(...)` (format binaire colonnaire, rapide et compact), `pd.read_html(url)`, `pd.read_clipboard()`.

## C.5 Exploration des données

Les premiers gestes devant tout nouveau jeu de données. Nous travaillons avec un jeu cohérent, reconstruit ici :

```python
rng = np.random.default_rng(0)
n = 300
prix_ref = {"Riz": 650, "Huile": 1200, "Savon": 400, "Sucre": 800, "Lait": 950}
produits = rng.choice(list(prix_ref), n)
ventes = pd.DataFrame({
    "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "produit": produits,
    "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], n, p=[0.2, 0.4, 0.2, 0.2]),
    "quantite": rng.integers(1, 20, n),
})
ventes["prix_unitaire"] = ventes["produit"].map(prix_ref)
ventes["ca"] = ventes["quantite"] * ventes["prix_unitaire"]
```

## C.6 `head`, `tail`, `info`, `describe`, `shape`

```python
print(ventes.head())        # 5 premières lignes (head(10) pour 10)
print(ventes.tail(3))       # 3 dernières
print(ventes.shape)         # (300, 6)
ventes.info()               # types, valeurs non nulles, mémoire
print(ventes.describe())    # statistiques des colonnes NUMÉRIQUES
print(ventes.describe(include="object"))   # colonnes texte : count, unique, top, freq
print(ventes.dtypes)
print(ventes["produit"].value_counts())    # effectifs par modalité
print(ventes["region"].value_counts(normalize=True).round(3))   # proportions
print(ventes["produit"].nunique(), ventes["produit"].unique())
print(ventes.isna().sum())  # valeurs manquantes par colonne
print(ventes.sample(3, random_state=0))   # 3 lignes au hasard
```

Lecture de `describe()` : `count` (valeurs non nulles), `mean`, `std`, `min`, `25%/50%/75%` (quartiles), `max`. Comparez `mean` et `50%` : un grand écart signale une distribution asymétrique ou des outliers.

| Méthode | Question à laquelle elle répond |
|---|---|
| `shape` | combien de lignes/colonnes ? |
| `info()` | types corrects ? valeurs manquantes ? mémoire ? |
| `describe()` | échelle, dispersion, anomalies ? |
| `value_counts()` | répartition d'une catégorie |
| `isna().sum()` | où sont les trous ? |
| `nunique()` | cardinalité (identifiant ? catégorie ?) |
| `duplicated().sum()` | doublons ? |

> 💡 **Routine d'exploration** : `shape` → `head` → `info` → `describe` → `isna().sum()` → `duplicated().sum()` → `value_counts` sur les catégories. À faire à **chaque** nouveau jeu de données.

## C.7 Indexation avec `loc` et `iloc`

| | `loc` | `iloc` |
|---|---|---|
| Sélection par | **étiquettes** | **positions** (entiers) |
| Borne de fin dans une tranche | **incluse** | **exclue** |
| Booléens | oui | oui (tableau, pas Series) |

```python
df = pd.DataFrame({
    "nom": ["Awa", "Koffi", "Ama", "Yao"],
    "age": [21, 23, 22, 24],
    "note": [15.5, 12.0, 18.0, 9.5],
}, index=["e1", "e2", "e3", "e4"])

print(df.loc["e2"])                      # une ligne (Series)
print(df.loc["e2", "note"])              # 12.0 : un scalaire
print(df.loc["e1":"e3", ["nom", "note"]])   # e1, e2, e3 INCLUS
print(df.iloc[0])                        # première ligne
print(df.iloc[0:2, 0:2])                 # lignes 0-1, colonnes 0-1 (fin exclue)
print(df.iloc[-1, -1])                   # 9.5
print(df.loc[df["note"] > 12, "nom"])    # filtre + colonne
df.loc["e2", "note"] = 13.0              # modification SÛRE
```

Accès rapides à un scalaire : `df.at["e2", "note"]` (étiquettes), `df.iat[1, 2]` (positions).

⚠️ **Erreur fréquente** : avec l'index par défaut `0, 1, 2…`, `df.loc[0:2]` renvoie **3 lignes** (0, 1, 2) alors que `df.iloc[0:2]` en renvoie 2. Après un filtre ou un tri, l'index n'est plus `0..n-1` : `loc[0]` peut alors ne pas exister (`KeyError`) alors que `iloc[0]` marche toujours.

## C.8 Sélection de colonnes

```python
df["nom"]                  # Series
df[["nom", "note"]]        # DataFrame (liste de colonnes → doubles crochets)
df.nom                     # raccourci : à éviter (échoue si espace, conflit avec méthode)
df.select_dtypes(include="number")       # colonnes numériques
df.filter(like="no")                     # colonnes dont le nom contient « no »
df.drop(columns=["age"])                 # toutes sauf age
```

⚠️ `df["nom", "note"]` (un seul crochet avec virgule) → `KeyError`. Il faut une **liste** : `df[["nom", "note"]]`.

## C.9 Filtrage conditionnel

```python
df[df["note"] >= 12]                         # lignes dont la note ≥ 12
df[(df["age"] > 21) & (df["note"] >= 12)]    # ET : & ; parenthèses obligatoires
df[(df["age"] < 22) | (df["note"] > 17)]     # OU : |
df[~(df["age"] > 22)]                        # NON : ~
df[df["nom"].isin(["Awa", "Yao"])]           # appartenance
df[df["note"].between(10, 16)]               # entre (bornes incluses)
df[df["nom"].str.startswith("A")]            # texte
df.query("note >= 12 and age < 24")          # syntaxe lisible
```

Avec le jeu de ventes :

```python
gros = ventes[(ventes["region"] == "Sud") & (ventes["ca"] > 10_000)]
print(len(gros), gros["ca"].mean().round(0))
```

⚠️ `and`/`or` ne fonctionnent pas : `ValueError: The truth value of a Series is ambiguous`. Utilisez `&`, `|`, `~` avec parenthèses.

## C.10 Tri des données

```python
ventes.sort_values("ca", ascending=False).head(5)           # top 5 des ventes
ventes.sort_values(["region", "ca"], ascending=[True, False])   # tri multiple
ventes.sort_index()                                         # tri sur l'index
ventes.nlargest(3, "ca")                                    # 3 plus grandes valeurs
ventes.nsmallest(3, "quantite")
ventes["ca"].rank(ascending=False)                          # rang
```

`sort_values` renvoie un **nouveau** DataFrame (sauf `inplace=True`, déconseillé). Réaffecter : `ventes = ventes.sort_values("ca")`. Pour repartir d'un index propre : `.reset_index(drop=True)`.

## C.11 Modification et création de colonnes

```python
ventes["ca_k"] = ventes["ca"] / 1000                      # nouvelle colonne calculée
ventes["gros"] = ventes["ca"] > 10_000                    # booléenne
ventes["mois"] = ventes["date"].dt.month                  # depuis les dates
ventes["produit_maj"] = ventes["produit"].str.upper()     # transformation de texte (nouvelle colonne)
ventes["categorie_ca"] = pd.cut(ventes["ca"], bins=[0, 3000, 10000, np.inf],
                                labels=["faible", "moyen", "élevé"])
ventes["classe"] = np.where(ventes["quantite"] >= 10, "lot", "détail")
```

`assign` pour enchaîner sans modifier l'original :

```python
resultat = (ventes
            .assign(ca_tva=lambda d: d["ca"] * 1.18,
                    semaine=lambda d: d["date"].dt.isocalendar().week)
            .query("ca_tva > 5000"))
```

Renommer :

```python
ventes_ren = ventes.rename(columns={"ca": "chiffre_affaires"})           # renvoie un NOUVEAU DataFrame (ventes est inchangé)
noms_propres = ventes.columns.str.lower().str.replace(" ", "_")          # normaliser tous les noms de colonnes
# ventes.columns = noms_propres                                         # pour les appliquer : réaffecter
```

Remplacer des valeurs :

```python
ventes["region_code"] = ventes["region"].replace({"Nord": "N", "Sud": "S", "Est": "E", "Ouest": "O"})
```

Réordonner : `ventes = ventes[["date", "produit", "region"]]` ou `df.insert(1, "nouvelle", valeurs)`.

⚠️ **Erreur fréquente — `SettingWithCopyWarning`** : modifier une extraction.

```python
# ✘ risqué
sous = ventes[ventes["region"] == "Sud"]
sous["x"] = 1             # avertissement : on modifie peut-être une copie
# ✔ propre
sous = ventes[ventes["region"] == "Sud"].copy()
sous["x"] = 1
# ✔ modifier l'original avec loc
ventes.loc[ventes["region"] == "Sud", "x"] = 1
```

Évitez aussi l'affectation chaînée `df[cond]["col"] = v` : elle ne fait rien d'utile. Utilisez `df.loc[cond, "col"] = v`.

## C.12 Suppression de colonnes et de lignes

```python
df = pd.DataFrame({"nom": ["Awa", "Koffi", "Ama", "Yao"], "age": [21, 23, 22, 24],
                   "note": [15.5, 12.0, np.nan, 9.5], "colonne": [1, 2, 3, 4], "autre": [5, 6, 7, 8]},
                  index=["e1", "e2", "e3", "e4"])
df = df.drop(columns=["age"])             # colonne(s)
df = df.drop(index=["e1"])                # ligne(s) par étiquette
df = df.drop(df[df["note"] < 10].index)   # lignes selon une condition
df = df.dropna(subset=["note"])           # lignes avec NA dans une colonne
df = df.drop_duplicates()
df = df.reset_index(drop=True)            # renumérote 0..n-1
del df["colonne"]                         # alternative (modifie en place)
popped = df.pop("autre")                  # supprime ET renvoie la colonne
```

Le plus souvent, on **conserve** plutôt que supprimer : `df = df[df["note"] >= 10]`.

## C.13 Valeurs manquantes

Pandas représente les valeurs manquantes par `NaN` (flottants), `NaT` (dates), `<NA>` (types « nullable »). 

```python
d = pd.DataFrame({
    "nom": ["Awa", "Koffi", None, "Yao", "Esi"],
    "age": [21, np.nan, 22, 24, 25],
    "note": [15.5, 12.0, np.nan, np.nan, 17.0],
})
print(d.isna())                 # masque booléen
print(d.isna().sum())           # nb de NA par colonne
print(d.isna().mean().round(2)) # proportion de NA par colonne
print(d.notna().all(axis=1))    # lignes complètes
```

**Stratégies de traitement**

| Stratégie | Code | Quand ? |
|---|---|---|
| Supprimer les lignes | `d.dropna()` | peu de NA, aléatoires |
| Supprimer selon colonnes | `d.dropna(subset=["note"])` | colonne essentielle |
| Supprimer les colonnes | `d.dropna(axis=1, thresh=...)` | colonne presque vide |
| Constante | `d["age"].fillna(0)` | 0 a un sens métier |
| Moyenne / médiane | `d["note"].fillna(d["note"].median())` | numérique |
| Mode | `d["nom"].fillna(d["nom"].mode()[0])` | catégorielle |
| Propagation | `d.ffill()`, `d.bfill()` | séries temporelles |
| Interpolation | `d["note"].interpolate()` | séries ordonnées |
| Par groupe | `groupby().transform(...)` | NA dépendant d'une catégorie |

```python
d["age"] = d["age"].fillna(d["age"].median())
d["note"] = d["note"].fillna(d["note"].mean())
d["nom"] = d["nom"].fillna("Inconnu")
```

Imputation par groupe (plus fine) :

```python
ventes_na = pd.DataFrame({
    "region": ["N", "N", "S", "S", "S"],
    "ca": [100., np.nan, 300., np.nan, 500.],
})
ventes_na["ca"] = ventes_na["ca"].fillna(ventes_na.groupby("region")["ca"].transform("mean"))
print(ventes_na)     # NaN du N → 100 ; NaN du S → 400
```

⚠️ **Erreurs fréquentes**
- Remplacer par la moyenne **sur l'ensemble** avant la séparation train/test en Machine Learning : c'est du *data leakage* (voir cours ML).
- Confondre `NaN` et `"NaN"`/`"?"`/`-999` : lisez-les avec `na_values=[...]` ou `replace`.
- Supprimer aveuglément : vous pouvez éliminer 80 % des données si les NA sont dispersés.
- Se demander **pourquoi** la valeur manque (hasard ? panne ? refus de répondre ?) — la réponse conditionne la bonne méthode.

## C.14 Nettoyage des données

Exemple d'un fichier « sale » :

```python
sale = pd.DataFrame({
    "Nom Client ": [" awa ", "KOFFI", "Ama", "ama ", "Yao"],
    "Âge": ["21", "23", "abc", "22", "-5"],
    "Ville": ["cotonou", "Cotonou", "COTONOU", "Porto Novo", "porto-novo"],
    "Montant": ["1 200,50", "850", "n/a", "2 300,00", "400"],
})
```

Étapes classiques :

```python
# 1. Noms de colonnes propres
sale.columns = (sale.columns.str.strip().str.lower()
                .str.replace(" ", "_").str.replace("â", "a"))

# 2. Texte : espaces, casse, variantes
sale["nom_client"] = sale["nom_client"].str.strip().str.title()
sale["ville"] = (sale["ville"].str.strip().str.title()
                 .str.replace("-", " ", regex=False))

# 3. Nombres : conversion robuste (erreurs → NaN)
sale["age"] = pd.to_numeric(sale["age"], errors="coerce")
sale["montant"] = (sale["montant"].str.replace(" ", "", regex=False)
                   .str.replace(",", ".", regex=False))
sale["montant"] = pd.to_numeric(sale["montant"], errors="coerce")

# 4. Valeurs aberrantes → NaN
sale.loc[~sale["age"].between(0, 120), "age"] = np.nan
print(sale)
```

Autres outils : `str.contains`, `str.extract(r"(\d+)")`, `str.split(expand=True)`, `df.replace`, `clip`, `mask`, `where`.

**Checklist de nettoyage**

1. Noms de colonnes homogènes.
2. Types corrects (nombres, dates, catégories).
3. Espaces et casse du texte.
4. Variantes d'une même modalité (« Cotonou »/« cotonou »).
5. Valeurs impossibles (âge négatif, date future).
6. Valeurs manquantes déguisées (`-999`, `"?"`, `"n/a"`).
7. Doublons.
8. Cohérence entre colonnes (date de fin ≥ date de début).

## C.15 Doublons

```python
d = pd.DataFrame({"id": [1, 2, 2, 3, 3], "val": ["a", "b", "b", "c", "d"]})
print(d.duplicated())                       # [F F T F F] (la 1re occurrence n'est pas marquée)
print(d.duplicated(keep=False))             # marque TOUTES les occurrences
print(d.duplicated(subset=["id"]).sum())    # 2 : doublons sur id seulement
print(d.drop_duplicates())                  # lignes entièrement identiques
print(d.drop_duplicates(subset="id", keep="last"))
```

⚠️ Ne supprimez pas mécaniquement : deux ventes identiques le même jour peuvent être légitimes. Demandez-vous quelle combinaison de colonnes forme une **clé unique** réelle.

## C.16 Conversion des types

```python
d = pd.DataFrame({"prix": ["10", "20", "x"], "date": ["2025-01-05", "2025/02/10", "pas une date"],
                  "cat": ["a", "b", "a"], "ok": ["oui", "non", "oui"]})

d["prix"] = pd.to_numeric(d["prix"], errors="coerce")        # x → NaN
d["date"] = pd.to_datetime(d["date"], errors="coerce", format="mixed")
d["cat"] = d["cat"].astype("category")                       # économise de la mémoire
d["ok"] = d["ok"].map({"oui": True, "non": False})
d["prix"] = d["prix"].astype("Int64")                        # entier avec NA possible (nullable)
print(d.dtypes)
```

| Cible | Outil |
|---|---|
| nombre | `pd.to_numeric(..., errors="coerce")` |
| date | `pd.to_datetime(...)` |
| catégorie | `.astype("category")` |
| texte | `.astype("string")` ou `.astype(str)` |
| booléen | `.map({...})` |
| entier avec NA | `.astype("Int64")` (I majuscule) |

`errors` : `"raise"` (par défaut) arrête sur la première erreur ; `"coerce"` convertit l'invalide en NaN ; à utiliser en examinant ensuite combien de lignes ont échoué.

## C.17 GroupBy

Principe **split – apply – combine** : couper en groupes, appliquer une fonction à chaque groupe, recombiner.

```
   ventes ── split par region ─▶ [Nord][Sud][Est][Ouest] ── apply sum ─▶ ── combine ─▶ Series
```

```python
print(ventes.groupby("region")["ca"].sum())
print(ventes.groupby("region")["ca"].mean().round(0))
print(ventes.groupby("produit").size())                  # nombre de lignes par groupe
print(ventes.groupby(["region", "produit"])["quantite"].sum().head())   # plusieurs clés
print(ventes.groupby("region")[["quantite", "ca"]].sum())      # plusieurs colonnes
```

Le résultat est indexé par la clé de groupe. Pour retrouver une colonne normale : `.reset_index()` ou `as_index=False`.

```python
par_region = ventes.groupby("region", as_index=False)["ca"].sum()
```

Itérer sur les groupes :

```python
for region, sous in ventes.groupby("region"):
    print(region, len(sous))
```

`transform` (même taille que l'entrée, valeur du groupe répétée) et `filter` :

```python
ventes["part_region"] = ventes["ca"] / ventes.groupby("region")["ca"].transform("sum")
grands = ventes.groupby("produit").filter(lambda g: g["ca"].sum() > 300_000)
```

## C.18 Agrégations

```python
resume = ventes.groupby("region").agg(
    nb_ventes=("ca", "size"),
    ca_total=("ca", "sum"),
    ca_moyen=("ca", "mean"),
    qte_max=("quantite", "max"),
    nb_produits=("produit", "nunique"),
)
print(resume.round(0))
```

La syntaxe **« agrégation nommée »** `nom=(colonne, fonction)` est la plus lisible. Fonctions disponibles : `sum, mean, median, min, max, count, size, std, var, first, last, nunique`, ou toute fonction personnalisée :

```python
def etendue(x):
    return x.max() - x.min()

ventes.groupby("produit").agg(etendue_ca=("ca", etendue),
                              q90=("ca", lambda s: s.quantile(0.9)))
```

Autres agrégations :

```python
ventes["ca"].agg(["sum", "mean", "std"])
ventes.agg({"ca": ["sum", "mean"], "quantite": "max"})
ventes.groupby("region")["ca"].describe()
```

| `size` | `count` |
|---|---|
| nombre de lignes du groupe (NA inclus) | nombre de valeurs **non nulles** de la colonne |

## C.19 Merge, Join et Concat

### `concat` : empiler ou juxtaposer

```python
janvier = pd.DataFrame({"produit": ["Riz", "Huile"], "ca": [100, 200]})
fevrier = pd.DataFrame({"produit": ["Riz", "Sucre"], "ca": [150, 80]})
tout = pd.concat([janvier, fevrier], ignore_index=True)                     # lignes
cote_a_cote = pd.concat([janvier, fevrier], axis=1, keys=["jan", "fev"])    # colonnes
```

### `merge` : jointure à la SQL

```python
clients = pd.DataFrame({"id_client": [1, 2, 3, 4], "nom": ["Awa", "Koffi", "Ama", "Yao"]})
commandes = pd.DataFrame({"id_cmd": [10, 11, 12, 13], "id_client": [1, 1, 3, 9], "montant": [50, 70, 20, 99]})

print(pd.merge(commandes, clients, on="id_client", how="inner"))   # clients présents des deux côtés
print(pd.merge(commandes, clients, on="id_client", how="left"))    # toutes les commandes
print(pd.merge(commandes, clients, on="id_client", how="right"))
print(pd.merge(commandes, clients, on="id_client", how="outer", indicator=True))
```

| `how` | Garde |
|---|---|
| `inner` | seulement les clés communes |
| `left` | toutes les lignes de gauche |
| `right` | toutes les lignes de droite |
| `outer` | toutes les lignes des deux |

Clés de noms différents : `left_on="id", right_on="client_id"`. Clés multiples : `on=["a", "b"]`. Suffixes en cas de colonnes homonymes : `suffixes=("_g", "_d")`.

Détecter les problèmes de jointure :

```python
pd.merge(commandes, clients, on="id_client", how="left", validate="many_to_one")  # lève une erreur si doublons côté clients
```

### `join` : jointure sur l'index

```python
a = clients.set_index("id_client")
b = commandes.set_index("id_client")
print(a.join(b, how="inner"))
```

⚠️ **Erreurs fréquentes**
- **Explosion de lignes** : clé en doublon des deux côtés → produit cartésien. Vérifiez `len(avant)` vs `len(après)` et utilisez `validate=`.
- Clés de types différents (`int` vs `str`) → aucune correspondance : alignez avec `astype`.
- Perte de lignes avec `inner` sans s'en apercevoir : comptez avant/après, ou utilisez `how="left"` + `indicator=True`.

## C.20 Pivot et `pivot_table`

**`pivot_table`** : tableau croisé dynamique (comme sous Excel).

```python
t = ventes.pivot_table(index="region", columns="produit", values="ca",
                       aggfunc="sum", fill_value=0, margins=True, margins_name="Total")
print(t.round(0))
```

- `index` : ce qui forme les lignes ; `columns` : ce qui forme les colonnes ; `values` : ce qu'on agrège ; `aggfunc` : comment (`"sum"`, `"mean"`, `["sum", "count"]`, dict…).

`pivot` : simple réorganisation **sans agrégation** (échoue si doublons).

```python
long = pd.DataFrame({"mois": ["jan", "jan", "fev", "fev"], "prod": ["A", "B", "A", "B"], "v": [1, 2, 3, 4]})
large = long.pivot(index="mois", columns="prod", values="v")
```

Opération inverse : `melt` (large → long).

```python
print(large.reset_index().melt(id_vars="mois", var_name="prod", value_name="v"))
```

`crosstab` (effectifs croisés) :

```python
print(pd.crosstab(ventes["region"], ventes["produit"], normalize="index").round(2))
```

`stack` / `unstack` : basculer un niveau d'index entre lignes et colonnes.

```python
g = ventes.groupby(["region", "produit"])["ca"].sum()
print(g.unstack().round(0))
```

## C.21 `apply`, `map` et fonctions personnalisées

| Méthode | Opère sur | Usage |
|---|---|---|
| `Series.map(dict/func)` | chaque élément d'une Series | recodage, dictionnaire |
| `Series.apply(func)` | chaque élément | fonction Python |
| `DataFrame.apply(func, axis=1)` | chaque ligne (ou colonne si `axis=0`) | logique multi-colonnes |
| `DataFrame.map(func)` (anciennement `applymap`) | chaque cellule | transformation globale |

```python
ventes["produit"].map({"Riz": "Alimentaire", "Huile": "Alimentaire", "Sucre": "Alimentaire",
                       "Lait": "Alimentaire", "Savon": "Hygiène"})

def statut(ligne):
    if ligne["ca"] > 15000 and ligne["region"] == "Sud":
        return "prioritaire"
    return "standard"

ventes["statut"] = ventes.apply(statut, axis=1)
```

**Performance** : `apply(axis=1)` est une boucle Python déguisée, **lente**. Comparez :

```python
import time
t = time.perf_counter(); r1 = ventes.apply(lambda r: r["quantite"] * r["prix_unitaire"], axis=1); t1 = time.perf_counter() - t
t = time.perf_counter(); r2 = ventes["quantite"] * ventes["prix_unitaire"];                         t2 = time.perf_counter() - t
print(f"apply {t1:.4f}s vs vectorisé {t2:.5f}s")
```

Hiérarchie de préférence : **opérations vectorisées** > `np.where`/`np.select`/`pd.cut` > `map` > `apply` > boucle `for`.

Style fonctionnel avec `pipe` :

```python
def retirer_petites_ventes(d, seuil):
    return d[d["ca"] >= seuil]

resultat = (ventes
            .pipe(retirer_petites_ventes, 5000)
            .groupby("produit")["ca"].sum()
            .sort_values(ascending=False))
```

## C.22 Manipulation des dates

```python
d = pd.DataFrame({"date": pd.to_datetime(["2025-01-15", "2025-02-28", "2025-12-31"])})
d["annee"] = d["date"].dt.year
d["mois"] = d["date"].dt.month
d["nom_mois"] = d["date"].dt.month_name()
d["jour_semaine"] = d["date"].dt.day_name()
d["trimestre"] = d["date"].dt.quarter
d["fin_de_semaine"] = d["date"].dt.dayofweek >= 5      # 0 = lundi
print(d)
```

Formats : `pd.to_datetime("15/02/2025", format="%d/%m/%Y")`. Codes : `%d` jour, `%m` mois, `%Y` année sur 4 chiffres, `%H:%M:%S` heure.

Calculs :

```python
d["delta"] = d["date"] - pd.Timestamp("2025-01-01")        # Timedelta
d["jours"] = d["delta"].dt.days
d["dans_30j"] = d["date"] + pd.Timedelta(days=30)
d["date"].dt.strftime("%d/%m/%Y")                          # vers texte
```

⚠️ `"03/04/2025"` est ambigu (3 avril ou 4 mars ?). Précisez toujours `format=` ou `dayfirst=True`.

## C.23 Séries temporelles

Placer la date en **index** ouvre un jeu d'outils puissant.

```python
ts = ventes.set_index("date").sort_index()
print(ts.loc["2025-03"].shape)                   # tout mars 2025 (.loc : indispensable sur un DataFrame)
print(ts.loc["2025-02-01":"2025-02-28", "ca"].sum())
```

**`resample`** : regroupement temporel (comme `groupby` pour les dates).

```python
mensuel = ts["ca"].resample("MS").sum()          # MS = début de mois
hebdo = ts["ca"].resample("W").sum()
print(mensuel.head(3))
```

Alias de fréquence : `D` jour, `W` semaine, `MS` début de mois, `ME` fin de mois, `QS` trimestre, `YS` année, `h` heure.

**Fenêtres glissantes** :

```python
quotidien = ts["ca"].resample("D").sum()
mm7 = quotidien.rolling(window=7).mean()         # moyenne mobile 7 jours
cumul = quotidien.cumsum()
evol = mensuel.pct_change() * 100                # variation d'un mois sur l'autre en %
```

**Décalages** : `quotidien.shift(1)` (valeur de la veille), `diff()` (différence avec la veille).

**Dates manquantes** : `quotidien.asfreq("D")` ou `reindex(pd.date_range(...), fill_value=0)` garantit une ligne par jour.

```python
idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
complet = ts["ca"].resample("D").sum().reindex(idx, fill_value=0)
print(len(complet))     # 365
```

🎯 **Utile quand** : ventes par semaine, fréquentation, capteurs, bourse, suivi de consommation.

## C.24 Exportation des données

```python
ventes.to_csv("data/ventes.csv", index=False, encoding="utf-8")
ventes.to_csv("data/ventes_fr.csv", index=False, sep=";", decimal=",")   # pour Excel français
ventes.to_excel("data/ventes.xlsx", index=False)
ventes.to_json("data/ventes.json", orient="records", date_format="iso", force_ascii=False)
ventes.to_parquet("data/ventes.parquet")        # nécessite pyarrow (pip install pyarrow)
```

Plusieurs feuilles Excel :

```python
with pd.ExcelWriter("data/rapport.xlsx") as w:
    resume.to_excel(w, sheet_name="Résumé")
    ventes.head(50).to_excel(w, sheet_name="Détail", index=False)
```

⚠️ Oublier `index=False` ajoute une colonne `Unnamed: 0` à la relecture. Le CSV perd les types (dates → texte) : utilisez `parse_dates` à la relecture, ou Parquet/pickle pour conserver les types.

## C.25 Optimisation des DataFrames

**1. Mesurer la mémoire**

```python
print(ventes.memory_usage(deep=True).sum() / 1024, "Ko")
```

**2. Réduire les types**

```python
opt = ventes.copy()
for c in ["produit", "region"]:
    opt[c] = opt[c].astype("category")
opt["quantite"] = pd.to_numeric(opt["quantite"], downcast="integer")
opt["prix_unitaire"] = pd.to_numeric(opt["prix_unitaire"], downcast="integer")
opt["ca"] = pd.to_numeric(opt["ca"], downcast="integer")
print(opt.memory_usage(deep=True).sum() / 1024, "Ko")
```

Une colonne texte à peu de modalités (région, produit) devient **catégorie** : gain mémoire fréquent de 90 %.

**3. Ne lire que le nécessaire**

```python
pd.read_csv("data/ventes.csv", usecols=["produit", "ca"], dtype={"produit": "category"})
```

**4. Lire par morceaux** (fichiers qui dépassent la mémoire)

```python
total = 0
for morceau in pd.read_csv("data/ventes.csv", chunksize=100):
    total += morceau["ca"].sum()
print(total)
```

**5. Vectoriser** (voir C.21), éviter `iterrows`, ne pas construire un DataFrame par `append` en boucle (collecter des listes, puis un seul `pd.concat`).

**6. Choisir un bon format** : Parquet est plusieurs fois plus petit et rapide que CSV.

## C.26 Bonnes pratiques et erreurs fréquentes

**Bonnes pratiques**

1. Explorer d'abord (C.6). Ne jamais supposer que les données sont propres.
2. Ne **jamais** modifier les données brutes : travailler sur une copie, écrire le résultat dans un autre fichier.
3. Chaîner les opérations (`.assign().query().groupby()`) pour un code lisible.
4. Vérifier les tailles avant/après chaque jointure ou filtre.
5. Utiliser `loc` pour affecter.
6. Préférer les types adaptés (category, datetime, Int64).
7. Écrire des fonctions de nettoyage réutilisables.
8. Garder les résultats intermédiaires vérifiables avec `assert` :

```python
assert ventes["quantite"].gt(0).all(), "quantité négative détectée"
assert ventes["produit"].isin(["Riz", "Huile", "Savon", "Sucre", "Lait"]).all(), "produit inconnu"
```

**Erreurs fréquentes** (résumé)

| Erreur / symptôme | Cause | Solution |
|---|---|---|
| `KeyError: 'col'` | faute de frappe, espaces dans le nom | `df.columns`, `str.strip()` |
| `SettingWithCopyWarning` | modification d'une extraction | `.copy()` / `.loc` |
| `truth value of a Series is ambiguous` | `and`/`or` | `&`, `|`, parenthèses |
| Résultat inattendu après filtre avec `loc[0]` | index non renuméroté | `reset_index(drop=True)` / `iloc` |
| Moyenne impossible : `TypeError` | colonne texte | `pd.to_numeric` |
| Lignes doublées après `merge` | clés dupliquées | `validate=`, `drop_duplicates` |
| `Unnamed: 0` | CSV sauvé avec index | `index=False` |
| Dates non reconnues | colonne `object` | `to_datetime` |
| Calcul extrêmement lent | `apply(axis=1)`, `iterrows` | vectoriser |
| `inplace=True` sans effet / chaîné | pratique déconseillée | réaffecter le résultat |

## NumPy et Pandas ensemble

Pandas est construit sur NumPy : chaque colonne est un tableau NumPy enrichi (index, étiquettes, valeurs manquantes, types étendus).

**Passage de l'un à l'autre**

```python
arr = ventes["ca"].to_numpy()                # Series → ndarray
mat = ventes[["quantite", "prix_unitaire"]].to_numpy()   # DataFrame → ndarray 2D
df_np = pd.DataFrame(mat, columns=["q", "p"])             # ndarray → DataFrame
```

**Les ufuncs NumPy fonctionnent sur Pandas** et conservent l'index :

```python
ventes["log_ca"] = np.log(ventes["ca"])
ventes["ca_norm"] = (ventes["ca"] - ventes["ca"].mean()) / ventes["ca"].std()
ventes["tranche"] = np.select([ventes["ca"] < 3000, ventes["ca"] < 10000], ["bas", "moyen"], "haut")
```

**Quand préférer l'un ou l'autre ?**

| Situation | Outil |
|---|---|
| Calcul matriciel, simulation, algèbre | NumPy |
| Données tabulaires hétérogènes, étiquetées | Pandas |
| Entrée d'un modèle scikit-learn | les deux (DataFrame accepté, converti en ndarray) |
| Très grandes matrices numériques homogènes | NumPy |

**Pipeline classique**

```
CSV → Pandas (nettoyage, jointures, agrégats) → .to_numpy() → NumPy/scikit-learn (modèles) → résultats → Pandas (analyse)
```

Les différences à mémoriser :

| | NumPy | Pandas |
|---|---|---|
| `std` par défaut | `ddof=0` | `ddof=1` |
| NaN dans `mean` | contamine (`nan`) | ignoré (`skipna=True`) |
| Indexation | positions | étiquettes (`loc`) + positions (`iloc`) |
| Alignement | par position | par **index** |

```python
x = np.array([1., np.nan, 3.])
print(x.mean(), pd.Series(x).mean())    # nan 2.0
```

## Exercices C

**Questions de compréhension**

1. Quelle est la différence entre `loc` et `iloc` ? Que renvoie `df.loc[1:3]` comparé à `df.iloc[1:3]` ?
2. Pourquoi faut-il des parenthèses dans `df[(df.a > 1) & (df.b < 5)]` ?
3. Qu'est-ce que le schéma split-apply-combine ?
4. Quelle différence entre `merge(how="left")` et `merge(how="inner")` ?
5. Pourquoi `apply(axis=1)` est-il à éviter quand une opération vectorisée existe ?
6. Quel est l'intérêt de convertir une colonne texte à peu de modalités en `category` ?

**Petits exercices** (sur le jeu `ventes` des sections C.5 et suivantes)

7. Affichez le top 5 des lignes par chiffre d'affaires, avec seulement les colonnes `date`, `produit`, `ca`.
8. Quelle région a le chiffre d'affaires moyen par vente le plus élevé ?
9. Construisez un tableau région × produit des quantités vendues, avec totaux.
10. Calculez le CA mensuel et le mois de plus forte croissance par rapport au précédent.
11. Ajoutez une colonne « tranche » (`faible`/`moyen`/`élevé`) selon des quantiles de `ca` avec `pd.qcut`.
12. Créez une table `objectifs` (région → objectif mensuel) et joignez-la pour calculer l'écart à l'objectif.
13. Nettoyez la série `["  12,5 ", "7", "n/a", "9,25"]` en floats.

*(Solutions en fin de document.)*

---

# PARTIE D — Erreurs classiques des débutants

Cette section rassemble les pièges les plus fréquents, avec la **méthode de diagnostic**.

## D.1 Lire un message d'erreur

Python affiche une *traceback* : lisez-la **de bas en haut**. La dernière ligne donne le type d'erreur et le message ; les lignes au-dessus indiquent où.

```
Traceback (most recent call last):
  File "x.py", line 5, in <module>
    df["Note"]
KeyError: 'Note'
```

| Erreur | Signification | Cause typique |
|---|---|---|
| `KeyError` | clé/colonne introuvable | faute, espace, casse |
| `ValueError` | valeur de forme/contenu inadaptée | reshape impossible, broadcast |
| `TypeError` | type d'opération incorrect | `"a" + 1`, moyenne d'un texte |
| `IndexError` | indice hors limites | `a[10]` sur 5 éléments |
| `AttributeError` | méthode/attribut inexistant | `df.nom.upper()` (il faut `.str.upper()`) |
| `NameError` | nom non défini | cellule non exécutée, faute |
| `FileNotFoundError` | fichier introuvable | mauvais chemin |

## D.2 Les 15 pièges, avec correctifs

1. **Oublier de réaffecter** : `df.drop(columns=["x"])` seul ne change rien. → `df = df.drop(columns=["x"])`.
2. **`inplace=True`** : retourne `None`. `df = df.dropna(inplace=True)` met `df = None` ! → ne pas l'utiliser.
3. **`and` au lieu de `&`** (C.9, B.16).
4. **Index confondu avec une colonne** après `groupby` : `reset_index()`.
5. **`loc` vs `iloc`** : bornes incluses ou non.
6. **Copie vs vue** : `.copy()` avant de modifier.
7. **Axes inversés** : `axis=0` (le long des lignes) vs `axis=1`.
8. **Type texte pour des nombres** : `df.dtypes` → `to_numeric`.
9. **Dates non converties** : `to_datetime`.
10. **NaN ignorés silencieusement par Pandas** : la moyenne est calculée sur les non-nuls ; vérifier `count`.
11. **Jointure qui multiplie les lignes** : comparer les effectifs, `validate=`.
12. **Division entière / flottante** : `7 // 2 = 3`, `7 / 2 = 3.5`.
13. **Aléatoire non reproductible** : fixer la graine.
14. **Notebook : état caché** : exécuter les cellules dans le désordre. → `Kernel > Restart & Run All` régulièrement.
15. **Écraser le fichier brut** : toujours sauvegarder le nettoyé sous un autre nom.

## D.3 Méthode de débogage en 5 étapes

1. **Reproduire** avec le plus petit exemple possible (3 lignes).
2. **Afficher** `type()`, `shape`, `dtype`, `head()` à chaque étape.
3. **Isoler** : exécuter l'expression par morceaux (`a`, puis `a.b`, puis `a.b.c`).
4. **Lire la documentation** : `help(pd.merge)`, `df.groupby?`.
5. **Chercher le message exact** dans la documentation officielle (pandas.pydata.org, numpy.org/doc).

---

# Solutions des exercices du cours

*(Tentez d'abord les exercices !)*

## Solutions A

```python
# 1
print(type(3 / 2), 3 / 2)     # float 1.5
print(type(3 // 2), 3 // 2)   # int 1
print(type(3 % 2), 3 % 2)     # int 1

# 2
print([n ** 2 for n in range(1, 16) if n % 2 == 1])

# 3
v = [0, 1, 2, 3, 4, 5]
print(v[1:-1], v[::-2])       # [1, 2, 3, 4] [5, 3, 1]

# 4 : b et a désignent le même objet (liste mutable) ; b.append agit sur cet objet unique.

# 5
def mention(note):
    if note >= 16: return "Très bien"
    if note >= 14: return "Bien"
    if note >= 12: return "Assez bien"
    if note >= 10: return "Passable"
    return "Ajourné"
print([mention(n) for n in [18, 14.5, 12, 10, 7]])
```

## Solutions B

Questions : **1.** `shape` = tailles par axe `(4,5,2)`, `size` = 40, `ndim` = 3. **2.** Formes `(3,)` et `(2,)` : 3 ≠ 2 ; `(3,1)` et `(2,)` → `(3,2)` valide. **3.** Vue : mêmes données (slicing, reshape, `.T`). Copie : données indépendantes (masque, fancy indexing, `.copy()`). **4.** Par convention IEEE 754, `nan` n'est égal à rien, pas même à lui-même ; on teste avec `np.isnan`. **5.** `arange` quand on connaît le pas, `linspace` quand on connaît le nombre de points et les bornes.

```python
import numpy as np

# 6
a = np.arange(0, 101, 5)
print(a[a % 3 == 0])                    # [ 0 15 30 45 60 75 90]

# 7
M = np.ones((5, 5), dtype=int)
M[1:-1, 1:-1] = 0
print(M)

# 8
M = np.arange(20).reshape(4, 5)
print(M[:, -1], M[[1, 3]], M[:2, -3:], sep="\n")

# 9
rng = np.random.default_rng(0)
ventes = rng.integers(10, 100, size=(6, 12))
print(ventes.argmax(axis=1) + 1)        # meilleur mois (1..12) de chaque produit
print(ventes.sum(axis=1).argmax())      # produit le plus vendu sur l'année

# 10
X = rng.normal(size=(100, 4))
Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print(Xn.min(axis=0), Xn.max(axis=0))   # [0 0 0 0] [1 1 1 1]

# 11
def prop_au_dessus_moyenne(v):
    v = np.asarray(v)
    return (v > v.mean()).mean()
print(prop_au_dessus_moyenne([10, 12, 14, 20]))   # 0.25 (seul 20 dépasse la moyenne 14)

# 12
P = rng.random((5, 3))
D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
print(D.shape, np.allclose(np.diag(D), 0))
```

## Solutions C

Questions : **1.** `loc` : étiquettes, fin incluse ; `iloc` : positions, fin exclue. `df.loc[1:3]` → 3 lignes, `df.iloc[1:3]` → 2 lignes. **2.** `&` a une priorité supérieure à `>` : sans parenthèses, Python évalue `1 & df.b` d'abord. **3.** Séparer en groupes, appliquer une fonction à chacun, recombiner les résultats. **4.** `left` garde toutes les lignes de gauche (NaN si pas d'appariement), `inner` ne garde que les clés communes. **5.** Elle exécute une boucle Python ligne par ligne, 10 à 1000 fois plus lente. **6.** Mémoire réduite, opérations plus rapides, et ordre/modalités explicites.

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
n = 300
prix_ref = {"Riz": 650, "Huile": 1200, "Savon": 400, "Sucre": 800, "Lait": 950}
ventes = pd.DataFrame({
    "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "produit": rng.choice(list(prix_ref), n),
    "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], n, p=[0.2, 0.4, 0.2, 0.2]),
    "quantite": rng.integers(1, 20, n),
})
ventes["prix_unitaire"] = ventes["produit"].map(prix_ref)
ventes["ca"] = ventes["quantite"] * ventes["prix_unitaire"]

# 7
print(ventes.nlargest(5, "ca")[["date", "produit", "ca"]])

# 8
print(ventes.groupby("region")["ca"].mean().idxmax())

# 9
print(ventes.pivot_table(index="region", columns="produit", values="quantite",
                         aggfunc="sum", margins=True, margins_name="Total"))

# 10
mensuel = ventes.set_index("date")["ca"].resample("MS").sum()
croissance = mensuel.pct_change()
print(croissance.idxmax().strftime("%B %Y"), round(croissance.max() * 100, 1), "%")

# 11
ventes["tranche"] = pd.qcut(ventes["ca"], q=3, labels=["faible", "moyen", "élevé"])
print(ventes["tranche"].value_counts())

# 12
objectifs = pd.DataFrame({"region": ["Nord", "Sud", "Est", "Ouest"], "objectif_mensuel": [40_000, 90_000, 40_000, 40_000]})
ventes["mois"] = ventes["date"].dt.to_period("M")
ca_rm = ventes.groupby(["region", "mois"], as_index=False)["ca"].sum()
ca_rm = ca_rm.merge(objectifs, on="region", how="left", validate="many_to_one")
ca_rm["ecart"] = ca_rm["ca"] - ca_rm["objectif_mensuel"]
print(ca_rm.head())

# 13
s = pd.Series(["  12,5 ", "7", "n/a", "9,25"])
print(pd.to_numeric(s.str.strip().str.replace(",", ".", regex=False), errors="coerce"))
```

---

# Aide-mémoire final

**NumPy**

| Tâche | Code |
|---|---|
| Créer | `np.array`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace` |
| Forme | `a.shape`, `a.reshape(-1, 1)`, `a.T`, `a.ravel()` |
| Sélection | `a[i, j]`, `a[:, j]`, `a[masque]`, `a[[1, 3]]` |
| Calcul | `+ - * / **`, `np.sqrt`, `np.exp`, `np.log` |
| Stats | `mean, median, std, var, min, max, sum` avec `axis=` |
| Conditions | `np.where`, `np.select`, `&`, `\|`, `~` |
| Tri | `np.sort`, `np.argsort`, `np.unique` |
| Assemblage | `concatenate`, `vstack`, `hstack`, `stack` |
| Algèbre | `@`, `np.linalg.solve`, `inv`, `eig`, `norm` |
| Aléatoire | `rng = np.random.default_rng(seed)` |

**Pandas**

| Tâche | Code |
|---|---|
| Lire / écrire | `read_csv`, `read_excel`, `to_csv(index=False)` |
| Explorer | `head, info, describe, shape, value_counts, isna().sum()` |
| Sélectionner | `df["c"]`, `df[["a","b"]]`, `loc`, `iloc` |
| Filtrer | `df[(cond1) & (cond2)]`, `query`, `isin`, `between` |
| Créer | `df["n"] = ...`, `assign`, `np.where`, `pd.cut` |
| Nettoyer | `dropna, fillna, drop_duplicates, astype, to_numeric, str.*` |
| Agréger | `groupby(...).agg(nom=("col","fonction"))` |
| Joindre | `merge(on=, how=, validate=)`, `concat`, `join` |
| Tableaux croisés | `pivot_table`, `crosstab`, `melt`, `unstack` |
| Dates | `to_datetime`, `.dt.*`, `set_index`, `resample`, `rolling` |

Passez maintenant aux notebooks : `01_DEBUTANT.ipynb` pour consolider, puis `02_INTERMEDIAIRE.ipynb` et `03_AVANCE.ipynb`.
