<div align="center">
  <a href="https://github.com/CNES/cars"><img src="docs/images/pictos/picto_cars.png" alt="CARS" title="CARS"  width="20%"></a>
  <h4> README CONFIGURATION DASK CARS-HAL  </h4>
</div>

Ce document contient la description de la configuration DASK utilisée par défaut avec CARS-HAL sur le cluster HAL.

La configuration est découpée en 3 fichiers:
- [dask.yaml](./dask.yaml) : configuration des éléments DASK de base
- [distributed.yaml](./distributed.yaml) : configuration du bloc distributed : scheduler et workers
- [jobqueue.yaml](./jobqueue.yaml) : configuration de l'interface DASK et du cluster PBS


## Configuration dask.yaml

[Lien documentation Configuration noyau DASK](https://docs.dask.org/en/stable/configuration-reference.html#dask)

| Option  | Valeur CARS | Valeur par défaut DASK | Description | Commentaires |
| ------- | ----------- | ---------------------- | ------------| ------------ |
| temporary-directory | null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#temporary-directory| Par défaut: tmpdir = dask-worker-space local |
| dataframe:shuffle-compression| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#dataframe.shuffle-compression | Par défaut:  a priori pas de compression. TODO: à tester si gain ? |
| array:svg:size | 120 | 120 | https://docs.dask.org/en/stable/configuration-reference.html#array.svg.size | Par défaut. |
| array:slicing:split-large-chunks| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#array.slicing.split-large-chunks | Par défaut.  TODO: peut etre à mettre à true pour limiter si trop de warnings. Peut etre pb de perf à tester.  |
| optimization:fuse:active| **true** | None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.active | **Activé** TODO: test l'impact on/off de la configuration ?  |
| optimization:fuse:ave-width| 1| 1 | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.ave-width |  Par défaut. **TODO : à tester/adapter pour optimiser plus. limit de width= num_nodes/height** |
| optimization:fuse:max-width| null| None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-width | Par défaut : dépendant de ave-width. 1.5 + ave_width * log(ave_width + 1) |
| optimization:fuse:max-height| .inf | inf | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-height | Par défaut : pas de limites max pour la valeur d'optimisation. Toutes possibilitées |
| optimization:fuse:max-depth-new-edges| null|None = null| https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-depth-new-edges| Par défaut :  dépendant de ave-width. ave_width * 1.5 |
| optimization:fuse:subgraphs| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.subgraphs | Par défaut: laisse l'optimisateur choisir. TODO: test?|
| optimization:fuse:rename-keys| true | true |https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.rename-keys |  Par défaut. TODO: test ?|

## Configuration distributed.yaml

[Lien documentation Configuration DASK Distributed ](https://docs.dask.org/en/stable/configuration-reference.html#distributed)

| Option  | Valeur CARS | Valeur par défaut DASK | Description | Commentaires |
| ------- | ----------- | ---------------------- | ------------| ------------ |
| version | **2** | NA| pas défini dans la documentation | Par défaut dans exemple. TODO: a garder ? pourquoi le spécifier ?  pas de documentation...   |
| logging:distributed| **info** | info |Exemple: https://docs.dask.org/en/latest/configuration.html et https://docs.dask.org/en/latest/debugging.html#logs| Niveau de log du bloc DASK distributed. *A adapter suivant le niveau de log de sortie. Mettre à debug pour niveau max* |
|  logging:distributed.client| warning | warning | https://docs.dask.org/en/latest/debugging.html#logs | Pour voir coté client seulement, à adapter pour analyser plus finement |
| logging:distributed.worker| debug | not set |https://docs.dask.org/en/latest/debugging.html#logs | Par défaut. Pour beaucoup plus d'infos coté workers : à décommenter.|
| logging:bokeh | **critical** | error ?  |  https://docs.dask.org/en/latest/configuration.html | Limitation des logs afin de limiter les warnings/erreurs intempestives.  |
| logging:tornado | **critical** | warning ? | https://docs.dask.org/en/latest/configuration.html | dupliquer dans la conf de sortie. A tester dans dask.yaml. Comprend pas le lien direct avec http://stackoverflow.com/questions/21234772/python-tornado-disable-logging-to-stderr. L'idée est de supprimer les messages de tornado mais le lien ne parle pas de la conf dask|
| logging:tornado.application | **error** | ? | https://distributed.dask.org/en/latest/develop.html#tornado https://www.tornadoweb.org/en/stable/gen.html http://www.tornadoweb.org/en/stable/ioloop.html  http://www.tornadoweb.org/en/stable/networking.html | A creuser pour analyser les perfs de tornado (cf timeout TCP plus haut...), il faudrait pouvoir monter le niveau de logs sur tornado. TODO : A tester précisément  |
| scheduler:blocked-handlers|  [] | [] | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.blocked-handlers | Par défaut. |
| scheduler:allowed-failures| **10** | 3 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.allowed-failures |Augmentation pour passer les cas. TODO: il faudrait baisser cette valeur, semble un pis allé car il faudrait savoir pourquoi le worker stop.  |
| scheduler:bandwidth| **100000** | 100000000 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.bandwidth | Valeur testé mise plus basse : 100mbps vs 100G ? débit estimé. suffisant ? TODO: analyse fine d'utilisation de ce paramètre. |
| scheduler:transition-log-length| 100000 | 100000 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.transition-log-length | Par défaut. Unité? TODO: Taille de stockage des logs tournants. Si la mémoire ne bloque pas, semble ok. a baisser si besoin d'optimiser ? |
| scheduler:work-stealing| True | True | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.work-stealing | Par défaut. load balancing des taches entre workers. TODO: Peut etre dangereux sur la stabilité avec des images ? |
| worker:blocked-handlers|  [] | [] | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.blocked-handlers | Par défaut  |
| :worker:multiprocessing-method| **forkserver** | spawn  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.multiprocessing-method| Modification testée. TODO:  est ce qu'on gagne avec forkserver. Perf vs stabilité ... spawn vs fork vs forkserver. semble ok avec forkserver comme solution intermédiaire mais peut etre pas si lent avec spawn et plus stable vu les empilements de couche de logiciels sur chaque worker multiprocessé.   |
| worker:connections:outgoing et incoming| 50 et **50**  | 50 et 10 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.connections.outgoing https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.connections.incoming | Par défaut pour outgoing. Modifié à 50 pour incoming.  TODO: analyse précise des connections en cours pour savoir. Bien de ne pas saturer. |
| worker:validate| **true**  | false | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.validate |  Augmentation pour debug par défaut.  TODO: à voir l'impact? |
| worker:lifetime:restart | **True** | False | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.restart | Commenté. TODO: à tester. qu'est ce que deadline? à maitriser. J'enleverais l'option ou remettrait à false par défaut (je mets ca dans la MR), ou il faut analyser finement l'utilisation du lifetime "Lifetime was intended to provide a mechanism for a periodic reset of workers, such as might be useful when dealing with libraries that might leak memory and so benefit from cycling worker processes. It is orthogonal to adaptive." mrocklin https://github.com/dask/distributed/issues/3141 . Est ce que le lifetime serait à  caler sur le walltime ? |
| worker:lifetime:duration| null  | None=null | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.duration | Par défaut. |
| worker:lifetime:stagger| 0  | 0 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.stagger | Par défaut. |
| worker:profile:interval| 10 ms  | 10ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.profile.interval | Par défaut. |
| worker:profile:cycle| 1000 ms  | 1000ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.profile.cycle | Par défaut. |
| worker:memory:target| 0.60  | 0.6 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.target | Par défaut.  |
| worker:memory:spill| 0.70  | 0.7 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.spill | Par défaut.  |
| worker:memory:pause| 0.80  | 0.8 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.pause | Par défaut.  |
| worker:memory:terminate| 0.95 | 0.95 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.terminate | Par défaut.  |
| comm:retry:count| **10** | 0 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.retry.count | Testé pour améliorer la stabilité. TODO: par défaut: 0 devrait pas avoir à changer ca. pour l'instant on garde à 10 testé et commenté. Mais cela montre un soucis de communication plus bas entre workers |
| comm:compression| auto | auto  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.compression | Par défaut. |
| comm:default-scheme| tcp  | tcp | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.default-scheme | Par défaut. |
| comm:socket-backlog| 2048  | 2048 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.socket-backlog | Par défaut.  TODO: A priori suffisante mais ? |
| comm:timeouts:connects| **60s** | 10s  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.timeouts.connect | Augmentation de la valeur par défaut suite à tests. TODO: cela montre un soucis avant. On ne devrait pas y toucher ... |
| comm:timeouts:tcp| **120s** | 30s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.timeouts.tcp  | Augmentation suite à tests. TODO:  cela montre un soucis avant. On ne devrait pas y toucher. |
| comm:require-encryption| **False** | None=null | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.require-encryption  | Désactivation.  Pas besoin |
| dashboard:export-tool| False | False  |  pas documenté: https://docs.dask.org/en/stable/configuration-reference.html#distributed.dashboard.export-tool   | Par défaut |
| admin:tick:interval | 20 ms | 20 ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.tick.interval | Par défaut. |
| admin:tick:limit| **1s** | 3s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.tick.limit |  TODO: test à faire: par défaut 3s, on garde 1s ? |
| admin:log-format|  %(asctime)s :: %(name)s - %(levelname)s - %(message)s' | %(name)s - %(levelname)s - %(message)s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.log-format | défaut ok |
| admin:pdb-on-err| False| False | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.pdb-on-err | Par défaut. TODO test : a mettre à True pour debug. |

## Configuration jobqueue.yaml

La configuration principale se fait [dans le code  de CARS](https://gitlab.cnes.fr/3d/cars/-/blob/master/cars/cluster/dask.py#L307)

Mais également dans le fichier bien commenté [jobqueue.yaml](./jobqueue.yaml) si CARS ne change pas la configuration.
[Lien Documentation Configuration jobqueue](https://jobqueue.dask.org/en/latest/configuration.html)

## Configurations officielles : 


Des configurations de référence issues des projets DASK Github sont stockées dans le répertoire [reference_confs](./reference_confs/) pour exemple en date du 10 janvier 2022. 

Voici les liens pour dask, distributed et jobqueue des YAML par défaut et de leurs schémas associées: 

### Dask

- default: https://raw.githubusercontent.com/dask/dask/main/dask/dask.yaml
- schema: https://raw.githubusercontent.com/dask/dask/main/dask/dask-schema.yaml

### Distributed

- default: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed.yaml
- schema: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed-schema.yaml

### Jobqueue

- default: https://raw.githubusercontent.com/dask/dask-jobqueue/main/dask_jobqueue/jobqueue.yaml