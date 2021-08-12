============
CARS plugins
============

A plugin is a software component that adds a specific feature to an existing computer program.

For CARS, the goal is then to be able to add new functions between steps.

The difficulty is the optimization vs the modularity of the code.

Design:

- `Hooks <https://en.wikipedia.org/wiki/Hooking>`_ could be added between pipelines steps.
- Plugin could add external functionalities using the hooks API. (Example new cloud filtering after triangulation )
- Static cluster scheduling could be defined and added in parallel with the plugin

A step is a set of functional algorithms considered as a 3D block in CARS 3D pipelines.
They operate only on one tile (no cluster scheduling approach considered).

The current steps are

- Rectification:

    - generate_grid()

- Matching:

- Triangulation:

    - triangulate()
    - ...

- Filter cloud:
    -

- Rasterize:

    - rasterization()


For example, the devibration in CARS could be adapted as a plugin (or severals?)


Questions:

- plugins in prepare steps functions?
