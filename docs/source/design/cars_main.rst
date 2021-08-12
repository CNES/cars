===================
CARS main algorithm
===================

Algo:
- cars.conf.init_conf()
- cars.run(input, output, conf)

    - client, cluster = cars.cluster.init_cluster() %
    - graph = cars.cluster.init_graph(mode) %
    - asynchron writing launch ? pipeline write only on nodes by default ?
    - cars.pipelines.prepare(input, output, \*params?, cluster, graph)
    - asynchron writing launch ? pipeline write only on nodes by default ?
    - cars.pipelines.compute(input, output, \*params?, cluster, graph)
        - output can be point_cloud or dsm depending on dag

Questions:
- write_point_cloud or write_dsm ? sub step in parallel ?
