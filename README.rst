``bread`` -- tracking and determining lineage of budding yeast cells
====================================================================

Installation
------------

Recommended : create a virtual environment :

::

	conda create -n lpbs_bread python=3.7
	conda activate lpbs_bread
	python -m pip install pip --upgrade

Direct installation (requires `git`)

::

	# core+data dependencies
	python -m pip install git+https://github.com/ninivert/bread.git[data]

For development :

::

	# Install the package in development mode (installs dependencies and creates symlink)
	git clone https://github.com/ninivert/bread
	# alternatively, download and extract the zip of the repository
	cd bread
	pip install -e .


Command-line interface
----------------------

See ``docs/source/examples/lineage/demo.ipynb``

TODO
----

- ``bread.vis`` package for generating debug figures
- documentation
- refactor tracking with GNN
- implement nearest neighbours for lineage
- implement GUI
