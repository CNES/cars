# CARS Dask configuration

This file contains the description of DASK configuration used with CARS

This configuration is split into 3 files:
- [dask.yaml](./dask.yaml) : basic DASK configuration
- [distributed.yaml](./distributed.yaml) : configuration of DASK distributed : scheduler and workers
- [jobqueue.yaml](./jobqueue.yaml) : configuration of DASK and PBS cluster interface


##  dask.yaml configuration


| Option  | CARS value  | Default Dask value  | Comments |
| ------- | ----------- | ------------------- |  -------- |
| temporary-directory | null | None = null | Default: tmpdir = dask-worker-space local |
| dataframe:shuffle-compression| null | None = null  | Default:  no compression . TODO: test if gain |
| array:svg:size | 120 | 120  | Default. |
| array:slicing:split-large-chunks| null | None = null | Default.  TODO: set to true if too much warnings. Test if adds perfomance issues |
| optimization:fuse:active| **true** | None = null | **Activated** TODO: test impact on configuration activation  |
| optimization:fuse:ave-width| 1| 1  |  Default. **TODO : to test/adapt limit of width= num_nodes/height** |
| optimization:fuse:max-width| null| None = null  | Default : depends on de ave-width. 1.5 + ave_width * log(ave_width + 1) |
| optimization:fuse:max-height| .inf | inf  | Default : No max limitation. All possibilities. |
| optimization:fuse:max-depth-new-edges| null|None = null | Default :  depends on ave-width. ave_width * 1.5 |
| optimization:fuse:subgraphs| null | None = null | Default: Let optimizer choose. TODO: test?|
| optimization:fuse:rename-keys| true | true  |  Default. TODO: test ?|

## Configuration distributed.yaml


| Option  | CARS value  | Default Dask value  | Comments |
| ------- | ----------- | ------------------- | -------- |
| version | **2** | NA| Not defined in documentation | Default in example. TODO: keep ? why specify ?  no documentation...   |
| logging:distributed| **info** | info |  DASK distributed log level. *Adapt depending the output loglevel. Set to debug for mac level* |
|  logging:distributed.client| warning | warning | See for Client, adapt of refined analysis |
| logging:distributed.worker| debug | not set  | Default. For more workers infors : uncomment.|
| logging:bokeh | **critical** | error ?  |  Log limitation on warnings / errors polluting output.  |
| logging:tornado | **critical** | warning ?  | Duplicate in output configuration .  To test in dask.yaml. Link with  http://stackoverflow.com/questions/21234772/python-tornado-disable-logging-to-stderr ?. The idea is to delete tornado messages, but does not talk about dask conf|
| logging:tornado.application | **error** | ? | dig for more performance analysis of tornado (cf timeout TCP ...), We need to upgrade tornato loglevel. TODO |
| scheduler:blocked-handlers|  [] | []  | Default. |
| scheduler:allowed-failures| **10** | 3  | Augmented for some failed cases. TODO: lower this value |
| scheduler:bandwidth| **100000** | 100000000 | Tested values, set lower : 100mbps vs 100G ? estimated flow. enough ? TODO: analyse the use of this parameter. |
| scheduler:transition-log-length| 100000 | 100000 | Default. Unit? TODO: Storage size in spinning logs. Seems ok if memory doent freeze  |
| scheduler:work-stealing| True | True  | Default. load balancing of workers tasks. TODO: Could be dangerous for stability, working with images  |
| worker:blocked-handlers|  [] | []  | Default  |
| :worker:multiprocessing-method| **forkserver** | spawn  | Tested modification. TODO:  see the gain with forkserver. Perf vs stability ... spawn vs fork vs forkserver. Seems ok with forkserver  as intermediate solution but maybe not that slow with spawn, ans more stable, with software stacked on each multiprocessed worker.  |
| worker:connections:outgoing et incoming| 50 and **50**  | 50 and 10 | Default for outgoing. Modified to 50 for incoming.  TODO: Precise analysis of conections, current connections. Beware not to saturate. |
| worker:validate| **true**  | false |  Augmentation for debug Default.  TODO: see impact? |
| worker:lifetime:restart | **True** | False | Commented. TODO: to test. What is deadline ? See lifetime utilisation. "Lifetime was intended to provide a mechanism for a periodic reset of workers, such as might be useful when dealing with libraries that might leak memory and so benefit from cycling worker processes. It is orthogonal to adaptive." mrocklin https://github.com/dask/distributed/issues/3141 . Does lifetime as to be set the same as walltime  ? |
| worker:lifetime:duration| null  | None=null  | Default. |
| worker:lifetime:stagger| 0  | 0  | Default. |
| worker:profile:interval| 10 ms  | 10ms | Default. |
| worker:profile:cycle| 1000 ms  | 1000ms  | Default. |
| worker:memory:target| 0.60  | 0.6 | Default.  |
| worker:memory:spill| 0.70  | 0.7  | Default.  |
| worker:memory:pause| 0.80  | 0.8  | Default.  |
| worker:memory:terminate| 0.95 | 0.95  | Default.  |
| comm:retry:count| **10** | 0  |  Tested for stability improvement. TODO: Default: 0 should not change that. For now we keep it to 10. May be lower comminuration issue between workers |
| comm:compression| auto | auto   | Default. |
| comm:default-scheme| tcp  | tcp  | Default. |
| comm:socket-backlog| 2048  | 2048  | Default.  TODO: not sufficient ? |
| comm:timeouts:connects| **60s** | 10s   | Augmentation of value following done tests. TODO: issues before. Do not touch it ... |
| comm:timeouts:tcp| **120s** | 30s  |  Augmentation of value following done tests. TODO: issues before. Do not touch it ...  |
| comm:require-encryption| **False** | None=null  | Deactivated. No need |
| dashboard:export-tool| False | False   | Default |
| admin:tick:interval | 20 ms | 20 ms | Default. |
| admin:tick:limit| **1s** | 3s |  TODO: test it : default 3s, keep 1s ? |
| admin:log-format|  %(asctime)s :: %(name)s - %(levelname)s - %(message)s' | %(name)s - %(levelname)s - %(message)s | default ok |
| admin:pdb-on-err| False| False | Default. TODO test : set to true for debug. |

## Configuration jobqueue.yaml

The main configuration is done with [dask usage in  CARS](https://github.com/CNES/cars/blob/master/cars/orchestrator/cluster/pbs_dask_cluster.py)

But also in documented file [jobqueue.yaml](./jobqueue.yaml) if not overloaded in CARS.
[Jobqueue documentation](https://jobqueue.dask.org/en/latest/configuration.html)

## Official configurations : 

Reference dask configuration of dask projects are stored in [reference_confs](./reference_confs/)

### Dask

- default: https://raw.githubusercontent.com/dask/dask/main/dask/dask.yaml
- schema: https://raw.githubusercontent.com/dask/dask/main/dask/dask-schema.yaml

### Distributed

- default: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed.yaml
- schema: https://raw.githubusercontent.com/dask/distributed/main/distributed/distributed-schema.yaml

### Jobqueue

- default: https://raw.githubusercontent.com/dask/dask-jobqueue/main/dask_jobqueue/jobqueue.yaml