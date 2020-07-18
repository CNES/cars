You'll find here the contributing guide of the **CARS** project.

1. [Bug report](# Bug report)
2. [Coding guide](# Coding guide)
3. [Merge request acceptation process](# Merge request acceptation process)

# Bug report

Any proven or suspected malfunction should be traced in a bug report, the latter being an issue in the CARS github repository. Don't hesitate to do so: It is best to open a bug report and quickly resolve it than to let a problem remains in the project.

In the problem description, be as accurate as possible. Include:
* The procedure used to initialize the environment
* The incriminated command line or python function
* The content of the input and output configuration files (`content.json`)

**Notifying the potential bugs is the first way for contributing to a software.**

# Coding guide

Here are some rules to apply when developing a new functionality:
* Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* If possible, limit the use of classes as much as possible and opt for a functional approach. The classes are reserved for data modelling if it is impossible to do so using `xarray`.
* Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* All functions shall be documented (object, parameters, return values).
* Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the cars python modules.
* If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* Do not add new dependencies unless it is absolutely necessary, and only if it has a permissive license.
* Use the type hints provided by the `typing` python module.

Any code modification requires a Merge Request. It is forbidden to push patches directly into master (this branch is protected).

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add `WIP:` before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding `Closes xx` where xx is the reference number of the issue.

Likewise, if you work on a branch (which is recommended), prefix the branch's name by `xx-` in order to link it to the xx issue.

# Merge request acceptation process

The Merge Request will be merged into master after being review by a CARS steering committee (core committers) composed of:
* David Youssefi (CNES DNO/OT/IS)
* Emmanuelle Sarrazin (CNES DSO/SI/2A)
* Julien Michel (CNES CESBIO)
* Fabrice Buffe (CNES DSO/SI/QI)
* Aurélie Emilien (CS Group)

Only the members of this committee can merge into master.

The checklist of a Merge Request acceptance is the following:
* [ ] At least two code reviews have been done by members of the group above (who check among others the correct application of the rules listed in the [Coding guide](# Coding guide)).
* [ ] All comments of the reviewers has been dealt with and are closed
* [ ] The reviewers have signaled their approbation (thumb up)
* [ ] No developer is against the Merge Request (thumb down)
