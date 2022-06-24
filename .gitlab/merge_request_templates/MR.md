**A choisir dans le template les éléments à garder suivant le développement à faire.**

Si la Merge Request est en cours de développement, merci d'ajouter le mot clé `WIP` ou `Draft` afin d'éviter la fusion de cette dernière.

#### Résumé de la proposition

1. Que fait la MR ? (bien etre explicite sur le titre)
2. Liste des taches de la MR qui répondent à l'issue (partiellement ou globalement suivant si Closes ou Relates)
3. Etat du Reste à faire à reprendre pour finir l'issue
4. Lien vers l'issue source (Closes si la MR ferme l'issue, Relates si en fait une partie)

A choisir:
Closes #num_issue
Relates #num_issue

#### Détails techniques de l'implémentation

Cette section décrit les solutions techniques que la MR met en oeuvre et qui répondent à la description fonctionnelle de l'issue associée.

#### Stratégie de validation

En lien avec le plan de validation de l'issue, il faut décrire la stratégie de validation dont s'occupe la MR.
Au moins :
- [ ] Tests Unitaires (obligatoire)
- [ ] Tests Fonctionnels (intégration / interfaces avec d'autres outils)
- [ ] Tests Performances

Si besoin suivant l'issue/MR:
- [ ] Tests Visuels ? (doc)
- [ ] Tests Comparatifs ? (`feat` métier avec outil de référence)
  - dans ce cas citer les outils et leur paramétrage


#### Validation de la MR

Si la Merge Request est prête, merci de valider les étapes suivantes:
- [ ] mise à jour de la documentation Sphinx et vérification du résultat.
- [ ] tous les tests passent et la MR est couverte par la stratégie de validation
- [ ]  mise à jour du changelog Changelog.md
  - uniquement si la MR rempli l'un des objectifs suivants:
    - correction d'un bug
    - ajout d'une fonctionnalité métier
    - ajout d'une fonctionnalité à destination du grand public (communication)
  - suivre les recommandations de https://github.com/olivierlacan/keep-a-changelog/blob/master/CHANGELOG.md
    - inscrire la modification sous le titre `Unreleased`
- [ ] S'assurer qu'il y a toutes les infos dans la MR pour qu'un autre développeur puisse relire facilement, ce qu'on s'attendrait à avoir soi même pour relire la MR (cf résumé ci dessus)

#### Rappel Intégration Continue

Pour relancer l'intégration continue merci de laisser le commentaire suivant :
`Jenkins! Faster!! And you'd better make it work !`


