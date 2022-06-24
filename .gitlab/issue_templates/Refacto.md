Préconisations générales:
1. Pensez à bien mettre un **titre** d'issue explicite par rapport au domaine de refactoring/reconception.
2. Pensez à essayer au maximum d'aider un développeur qui devra reprendre ce travail ou relire avec des informations synthétiques les plus claires et précises

Il est important d'essayer de se mettre à la place de quelqu'un d'autre qui relira, ou dans la perspective d'une présentation à quelqu'un d'autre pour transférer le travail.

/label ~"Kind - Refacto"

### Contexte
Cette section donne le contexte général du refactoring prévu.

### Documentation API et fonctionnel

Cette partie décrit le fonctionnement du module/domaine visé par l'issue.

1. décrire bien ce que fait le bloc logiciel à haut niveau avec API utilisateurs et internes (nom fonctions, paramètres, noms de structures de données (classe ou autre) explicites )
2. décrire fonctionnellement les blocs avec le principe de base à haut niveau.

Des schémas UML sont toujours bien pour des classes.

Les objectifs:
- qu'un utilisateur sache exactement ce que fait le module par l'API
- qu'un autre développeur puisse comprendre/relire/reprendre le travail rapidement

### Plan de validation / tests

Cette section décrit la manière de tester le domaine fonctionnel du refactoring prévu.

En cohérence avec la documentation fonctionnel, cette section doit décrire **précisément** la façon de tester le module logiciel visé.

Il est important de considérer:
- les tests unitaires pour les fonctions de base
- les tests fonctionnels du  module : que doit on voir fonctionner pour que le module soit valide ?
- considérer le temps de calcul et séparer si des tests sont trop lourds

Utiliser les @pytest.mark pour organiser les tests suivant la découpe d'organisations des tests choisie.
