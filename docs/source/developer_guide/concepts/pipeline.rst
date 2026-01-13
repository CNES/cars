============
Pipeline
============

.. role:: raw-html(raw)
   :format: html

:raw-html:`<h1>Pipeline</h1>`


**Overview**


A *pipeline* is a series of steps of CARS 3D reconstruction framework.
It generates a new product (example: DSM) from another product (example: sensors) .



It is composed of:
* an pipeline factory that register all pipelines (including plugins)
* an abstract pipeline template
* Some subclass pipelines (surface_modeling, tie_points, merging, formatting, etc)

A pipeline use a global orchestrator to manage all its applications and datasets.
A series of applications are chained to process the data from input to output.


**Example**


Let's take an example of `tie_points` pipeline to describe the main steps:

First, we can notice that `tie_points` derives from `PipelineTemplate` and is registered with the decorator:

.. sourcecode:: python

    @Pipeline.register(
        "tie_points",
    )
    class TiePointsPipeline(PipelineTemplate):


Then,  pipeline name is defined in a subclass register, by is `short_name`, of `tie_points` pipeline.



In the `init` function, the pipeline is instantiated, with all the checks applied on the configuration file:

.. sourcecode:: python

        def __init__(self, conf, config_dir=None):

        # Used conf
        self.used_conf = {}

        # Transform relative path to absolute path
        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)

        # Check global conf
        self.check_global_schema(conf)
        ...



In the `run` function, the pipeline is executed, with all its applications processed in a row:

An orchestrator is instantiated at the beginning of the pipeline, and given to each application.

.. sourcecode:: python

    def run(self, log_dir=None, disp_range_grid=None):
        """
        Run pipeline

        """
        ...
        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=log_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as cars_orchestrator:
            # Run applications


