Release <numero_version>

/label ~"Kind - Release"

Liste de points à vérifier/faire pour la release en cours:

- [ ] Vérifier issues milestone (mettre lien milestone avec %nom_milestone)
- [ ] Finaliser Changelog de la version en cours: Vérifier en comparant avec les issues/MR du milestone de la version.
- [ ] Mise du tag sur la version finale après dernieres MR.
- [ ] Vérification Publication code sur github, read the docs, pypi
- [ ] Dernière Vérification si installation, tests, ...  (relance CI ou manuellement si pas automatique dans CI)
- [ ] Tests et Upgrade cars-hal
- [ ] Ajout dans module load si necessaire suivant projet
- [ ] Génération image docker et Publication du docker (pas automatique pour l'instant)
- [ ] Communication sur la release (si nécessaire mailing list ou autre)
