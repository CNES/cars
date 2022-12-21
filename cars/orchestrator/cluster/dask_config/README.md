<div align="center">
  <a href="https://github.com/CNES/cars"><img src="docs/images/pictos/picto_cars.png" alt="CARS" title="CARS"  width="20%"></a>
  <h4> README CONFIGURATION DASK CARS </h4>
</div>

This file contains the description of DASK configuration used with CARS

This configuration is split into 3 files:
- [dask.yaml](./dask.yaml) : basic DASK configuration
- [distributed.yaml](./distributed.yaml) : configuration of DASK distributed : scheduler and workers
- [jobqueue.yaml](./jobqueue.yaml) : configuration of DASK and PBS cluster interface


##  dask.yaml configuration

[DASK Documentation link](https://docs.dask.org/en/stable/configuration-reference.html#dask)

| Option  | CARS value  | Default Dask value  | Description | Comments |
| ------- | ----------- | ------------------- | ------------| -------- |
| temporary-directory | null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#temporary-directory| Default: tmpdir = dask-worker-space local |
| dataframe:shuffle-compression| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#dataframe.shuffle-compression | Default:  no compression . TODO: test if gain |
| array:svg:size | 120 | 120 | https://docs.dask.org/en/stable/configuration-reference.html#array.svg.size | Default. |
| array:slicing:split-large-chunks| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#array.slicing.split-large-chunks | Default.  TODO: set to true if too much warnings. Test if adds perfomance issues |
| optimization:fuse:active| **true** | None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.active | **Activated** TODO: test impact on configuration activation  |
| optimization:fuse:ave-width| 1| 1 | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.ave-width |  Default. **TODO : to test/adapt limit of width= num_nodes/height** |
| optimization:fuse:max-width| null| None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-width | Default : depends on de ave-width. 1.5 + ave_width * log(ave_width + 1) |
| optimization:fuse:max-height| .inf | inf | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-height | Default : No max limitation. All possibilities. |
| optimization:fuse:max-depth-new-edges| null|None = null| https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.max-depth-new-edges| Default :  depends on ave-width. ave_width * 1.5 |
| optimization:fuse:subgraphs| null | None = null | https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.subgraphs | Default: Let optimizer choose. TODO: test?|
| optimization:fuse:rename-keys| true | true |https://docs.dask.org/en/stable/configuration-reference.html#optimization.fuse.rename-keys |  Default. TODO: test ?|

## Configuration distributed.yaml

[Lien documentation Configuration DASK Distributed ](https://docs.dask.org/en/stable/configuration-reference.html#distributed)

| Option  | CARS value  | Default Dask value  | Description | Comments |
| ------- | ----------- | ------------------- | ------------| -------- |
| version | **2** | NA| Not defined in documentation | Default in example. TODO: keep ? why specify ?  no documentation...   |
| logging:distributed| **info** | info | Example: https://docs.dask.org/en/latest/configuration.html et https://docs.dask.org/en/latest/debugging.html#logs|  DASK distributed log level. *Adapt depending the output loglevel. Set to debug for mac level* |
|  logging:distributed.client| warning | warning | https://docs.dask.org/en/latest/debugging.html#logs | See for Client, adapt of refined analysis |
| logging:distributed.worker| debug | not set |https://docs.dask.org/en/latest/debugging.html#logs | Default. For more workers infors : uncomment.|
| logging:bokeh | **critical** | error ?  |  https://docs.dask.org/en/latest/configuration.html |  Log limitation on warnings / errors polluting output.  |
| logging:tornado | **critical** | warning ? | https://docs.dask.org/en/latest/configuration.html | Duplicate in output configuration .  To test in dask.yaml. Link with  http://stackoverflow.com/questions/21234772/python-tornado-disable-logging-to-stderr ?. The idea is to delete tornado messages, but does not talk about dask conf|
| logging:tornado.application | **error** | ? | https://distributed.dask.org/en/latest/develop.html#tornado https://www.tornadoweb.org/en/stable/gen.html http://www.tornadoweb.org/en/stable/ioloop.html  http://www.tornadoweb.org/en/stable/networking.html | dig for more performance analysis of tornado (cf timeout TCP ...), We need to upgrade tornato loglevel. TODO |
| scheduler:blocked-handlers|  [] | [] | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.blocked-handlers | Default. |
| scheduler:allowed-failures| **10** | 3 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.allowed-failures | Augmented for some failed cases. TODO: lower this value |
| scheduler:bandwidth| **100000** | 100000000 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.bandwidth | Tested values, set lower : 100mbps vs 100G ? estimated flow. enough ? TODO: analyse the use of this parameter. |
| scheduler:transition-log-length| 100000 | 100000 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.transition-log-length | Default. Unit? TODO: Storage size in spinning logs. Seems ok if memory doent freeze  |
| scheduler:work-stealing| True | True | https://docs.dask.org/en/stable/configuration-reference.html#distributed.scheduler.work-stealing | Default. load balancing of workers tasks. TODO: Could be dangerous for stability, working with images  |
| worker:blocked-handlers|  [] | [] | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.blocked-handlers | Default  |
| :worker:multiprocessing-method| **forkserver** | spawn  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.multiprocessing-method| Tested modification. TODO:  see the gain with forkserver. Perf vs stability ... spawn vs fork vs forkserver. Seems ok with forkserver  as intermediate solution but maybe not that slow with spawn, ans more stable, with software stacked on each multiprocessed worker.  |
| worker:connections:outgoing et incoming| 50 and **50**  | 50 and 10 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.connections.outgoing https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.connections.incoming | Default for outgoing. Modified to 50 for incoming.  TODO: Precise analysis of conections, current connections. Beware not to saturate. |
| worker:validate| **true**  | false | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.validate |  Augmentation for debug Default.  TODO: see impact? |
| worker:lifetime:restart | **True** | False | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.restart | Commented. TODO: to test. What is deadline ? See lifetime utilisation. "Lifetime was intended to provide a mechanism for a periodic reset of workers, such as might be useful when dealing with libraries that might leak memory and so benefit from cycling worker processes. It is orthogonal to adaptive." mrocklin https://github.com/dask/distributed/issues/3141 . Does lifetime as to be set the same as walltime  ? |
| worker:lifetime:duration| null  | None=null | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.duration | Default. |
| worker:lifetime:stagger| 0  | 0 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.lifetime.stagger | Default. |
| worker:profile:interval| 10 ms  | 10ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.profile.interval | Default. |
| worker:profile:cycle| 1000 ms  | 1000ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.profile.cycle | Default. |
| worker:memory:target| 0.60  | 0.6 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.target | Default.  |
| worker:memory:spill| 0.70  | 0.7 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.spill | Default.  |
| worker:memory:pause| 0.80  | 0.8 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.pause | Default.  |
| worker:memory:terminate| 0.95 | 0.95 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.worker.memory.terminate | Default.  |
| comm:retry:count| **10** | 0 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.retry.count |  Tested for stability improvement. TODO: Default: 0 should not change that. For now we keep it to 10. May be lower comminuration issue between workers |
| comm:compression| auto | auto  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.compression | Default. |
| comm:default-scheme| tcp  | tcp | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.default-scheme | Default. |
| comm:socket-backlog| 2048  | 2048 | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.socket-backlog | Default.  TODO: not sufficient ? |
| comm:timeouts:connects| **60s** | 10s  | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.timeouts.connect | Augmentation of value following done tests. TODO: issues before. Do not touch it ... |
| comm:timeouts:tcp| **120s** | 30s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.timeouts.tcp  |  Augmentation of value following done tests. TODO: issues before. Do not touch it ...  |
| comm:require-encryption| **False** | None=null | https://docs.dask.org/en/stable/configuration-reference.html#distributed.comm.require-encryption  | Deactivated. No need |
| dashboard:export-tool| False | False  |  pas documenté: https://docs.dask.org/en/stable/configuration-reference.html#distributed.dashboard.export-tool   | Default |
| admin:tick:interval | 20 ms | 20 ms | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.tick.interval | Default. |
| admin:tick:limit| **1s** | 3s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.tick.limit |  TODO: test it : default 3s, keep 1s ? |
| admin:log-format|  %(asctime)s :: %(name)s - %(levelname)s - %(message)s' | %(name)s - %(levelname)s - %(message)s | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.log-format | défaut ok |
| admin:pdb-on-err| False| False | https://docs.dask.org/en/stable/configuration-reference.html#distributed.admin.pdb-on-err | Default. TODO test : set to true for debug. |

## Configuration jobqueue.yaml

The main configuration is done with [dask usage in  CARS](https://github.com/CNES/cars/blob/master/cars/orchestrator/cluster/pbs_dask_cluster.py)

But also in documented file [jobqueue.yaml](./jobqueue.yaml) if not overloaded in CARS.
[Jobqueue documentation](https://jobqueue.dask.org/en/latest/configuration.html)

## Configurations officielles : 

Reference dask configuration of dask projects are stored in [reference_confs](./reference_confs/)  for 

### Dask

- default: https://raw.githubusercontent.com/dask/dask/main/dask/dask.yaml
- schema: https://raw.githubusercontent.com/dask/dask/main/dask/dask-schema.yaml

### Distributed

- default: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed.yaml
- schema: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed-schema.yaml

### Jobqueue

- default: https://raw.githubusercontent.com/dask/dask-jobqueue/main/dask_jobqueue/jobqueue.yaml