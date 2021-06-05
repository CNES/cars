#### Résumé de la proposition

#### Détails techniques de l'implémentation (si besoin)

#### Stratégie de validation

- [ ] Tests Unitaires ? (obligatoire pour du `feat` et `fix`)
- [ ] Tests Visuels ? (doc)
- [ ] Tests Fonctionnels ? (intégration / interfaces avec d'autres outils)
- [ ] Tests Comparatifs ? (`feat` métier avec outil de référence)
  - dans ce cas citer l'outil et son paramétrage 

#### MR non prête à merger

Si la Merge Request est en cours de développement, merci d'ajouter le mot clé `WIP` afin d'éviter la fusion de cette dernière.

#### MR prête à merger 

Si la Merge Request est prête, merci de valider les étapes suivantes:
- [ ] mise à jour de la documentation Sphinx et vérification du résultat. 
- [ ]  mise à jour du changelog Changelog.md
  - uniquement si la MR rempli l'un des objectifs suivants:
    - correction d'un bug
    - ajout d'une fonctionnalité métier
    - ajout d'une fonctionnalité à destination du grand public (communication)
  - suivre les recommandations de https://github.com/olivierlacan/keep-a-changelog/blob/master/CHANGELOG.md
    - inscrire la modification sous le titre `Unreleased`

#### Integration Continue

Pour relancer l'intégration continue merci de laisser le commentaire suivant :  
`Jenkins! Faster!! And you'd better make it work !`


