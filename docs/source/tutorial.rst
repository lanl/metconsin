Tutorial
============

This tutorial is meant to explain the example script provided in the ``Example`` directory. This example simulates and creates networks for a community of 10 organisms isolated from soil. Genome-scale models for the 10 organisms are provided in the 
``modelseedGems`` directory, and information about these models is contained in ``ModelSeed_info.csv``.

The tutorial includes setting a user-defined growth media, simulating and saving networks, computing some basic network statistics, and comparing networks.

Required Imports
-------------------

To complete the tutorial, we need to import the following modules:

.. code-block:: python

    import sys
    import os
    from metconsin import metconsin_sim,save_metconsin
    import pandas as pd
    from pathlib import Path
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import cobra as cb
    import contextlib
    import json

Additionally, we have to add the metconsin package to our path, if it is not there already. The following code 

.. code-block:: python

    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)



Setting a user defined growth media
------------------------------------------

User defined media *must* match up with the metabolite names or metabolite IDs, which we assume match up across models. In ``make_media.py``, we give an example of how to find the metabolite names used by the set of models, and create a growth media using the previously defined growth media saved in the model's ``.xml`` file (e.g. the media used in gap-filling by modelSEED). To do this,
we load the models using `cobrapy <https://opencobra.github.io/cobrapy/>`_ and inspect their set of reactions for the **EX_** tag, which indicates exchange reactions. We then use the set of reactants of these reactions as the set of metabolites the model can exchange. Note that this is the same way that :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>` identifies the 
set of exchanged metabolites for a model.

In our example, we use a ``.tsv`` file for the media, so that it can be easily opened and edited (e.g. in Microsoft Excel). 

.. warning::

    It is not uncommon for metabolite names to contain commas (,) so comma-separated files should be avoided.

It is simple to import the growth media using pandas and convert to a dictionary:

.. code-block:: python

    growth_media = pd.read_csv("growth_media.tsv",sep = '\t',index_col = 0).squeeze("columns").to_dict()


Setting metabolic uptake rate parameters
---------------------------------------------

Dynamic FBA requires some mapping from the environmental metabolites to a set of bounds on the exchange reaction. In this tutorial, we assume that lower bounds are constant, and upper bounds are simply linear in the amount of metabolite available. By defualt, 
MetConSIN will assume the constants of parameters of these linear functions are uniformly 1. However, we'd like to load in some parameters that we have perhaps fit to data (or, in this case, chosen at random from the interval :math:`[0.5:1.5]`). The parameters should
be passed as a dictionary keyed by the model names. Each entry in that dictionary can either be an array, ordered according to the model's ordering of the metabolites (which we probably don't want to try to figure out) or, more conveniently, a dictionary keyed by metabolite
names. Python dictionaries can be easily saved and loaded using the ``.json`` file format.

.. code-block:: python:

    with open("exchange_bounds.json") as fl:
        uptake_params = json.load(fl)


Running MetConSIN simulations
--------------------------------

To begin, we must tell MetConSIN where to find the GSM files. To do this, we use the ``ModelSeed_info.csv`` file, which contains a table with a **Species** column and a **File** column. We also need to create a list of the models we want to include
in the community as labeled in the **Species** column of model info file.

.. code-block:: python

    model_info_fl = "ModelSeed_info.csv"

    species = ['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']

Next, we create a directory for MetConSIN to save the results in. We also save the ``species`` list and the growth media in this file so that we can later confirm the conditions of the simulation.

.. code-block:: python

    tmlabel = dt.datetime.now()

    flder = "modelSeed_{}s_{}".format(len(species),tmlabel.strftime("%a%B%d_%Y_%H.%M"))

    Path(flder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(flder,"species.txt"),'w') as fl:
        fl.write("\n".join(species))
    with open(os.path.join(flder,"media.txt"),'w') as fl:
        fl.write("{}".format(growth_media))


Finally, we call :py:func:`metconsin_sim <metconsin.metconsin_sim>`, passing our growth media, how long we'd like the simulation to run for, as well as a choice of metabolic uptake bound functions.

.. code-block:: python

    with open("example.log",'w') as fl:
        metconsin_return = metconsin_sim(species,model_info_fl,endtime = 2,media = growth_media, ub_funs = "linearScale",linearScale=1.0,flobj = fl,resolution=0.01)

By default, MetConSIN prints a log of its activity. Here, we redirect this log to the file ``example.log`` by passing the file with the ``flobj`` parameter.

The results can be saved using the :py:func:`save_metconsin <metconsin.save_metconsin>` function:

.. code-block:: python

    flder = os.path.join(flder,"metconsin_results")

    save_metconsin(metconsin_return, flder)

:py:func:`save_metconsin <metconsin.save_metconsin>` saves the simulation dynamics in two tab-separated files: ``Microbes.tsv`` and ``Metabolites.tsv`` with rows corresponding to state variables (microbes or metabolites) and columns
corresponding to time-points. It also creates plots of the simulation dynamics (although these are not publication quality) and saves a list of times that the bases were changed for any microbe.

.. note::

    To-Do: Include which bases were changed at these times.

Finally, it creates a set of sub-directories to save internal and exchange fluxes, as well as the sequence of interaction networks.