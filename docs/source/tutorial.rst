Tutorial
============

This tutorial is meant to explain the example script ``Ten_Taxa_Example.py`` provided in the ``Example`` directory. This example simulates and creates networks for a community of 10 organisms isolated from soil. Genome-scale models for the 10 organisms are provided in the 
``modelseedGems`` directory, and information about these models is contained in ``ModelSeed_info.csv``. We also provide another example, ``Community_Comparison_Example.py`` that was used to test the method on various subsets of the 10 organisms for the tool publication. 

.. note:: 

    We also include a second example, ``Community_Comparison_Example.py``, which simply repeats the procedure with two subsets of the models used in ``Ten_Taxa_Example.py``.

The tutorial includes setting a user-defined growth media, simulating and saving networks, computing some basic network statistics, and comparing networks.

Required Imports
-------------------

To complete the tutorial, we need to import the following modules:

.. code-block:: python

    import sys
    import os
    import pandas as pd
    from pathlib import Path
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import seaborn as sb

Additionally, we have to add the metconsin package to our path, if it is not there already. The following code adds the current parent directory to the path, which is the 
``metconsin`` directory if run from the ``Example`` sub-directorie.

.. code-block:: python

    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    from metconsin import metconsin_sim,save_metconsin



Setting a user defined growth media
------------------------------------------

User defined media *must* match up with the metabolite names or metabolite IDs, which we assume match up across models. In ``make_media.py``, we give an example of how to find the metabolite names used by the set of models, and create a growth media using the previously defined growth media saved in the model's ``.xml`` file (e.g. the media used in gap-filling by modelSEED). To do this,
we load the models using `cobrapy <https://opencobra.github.io/cobrapy/>`_ and inspect their set of reactions for the **EX_** tag, which indicates exchange reactions. We then use the set of reactants of these reactions as the set of metabolites the model can exchange. Note that this is the same way that :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>` identifies the 
set of exchanged metabolites for a model.

In our example, we use a ``.tsv`` file for the media, so that it can be easily opened and edited (e.g. in Microsoft Excel). 

.. warning::

    It is not uncommon for metabolite names to contain commas, so comma-separated files should be avoided.

It is simple to import the growth media using pandas and convert to a dictionary:

.. code-block:: python

    growth_media = pd.read_csv("uniform_media.tsv",sep = '\t',index_col = 0).squeeze("columns").to_dict()

The growth media supplied simply assumes 100 units of each metabolite exchanged by any of the models is available. We can adjust the growth media by editing the resulting dictionary. Here, we might want to limit glucose.

.. code-block:: python 

    growth_media["D-Glucose_e0"] = 10

Alternatively, we could add some flow of metabolites. To simulate an aerobic environment, we can provide a constant flow of oxygen into the simulation. To do this, we simply create a dictionary of our desired flows, and give this to MetConSIN:

.. code-block:: python

    oxygen_in = {"O2_e0":100}


.. note::

    Metabolite names need to match the names of the exchanged metabolite stored in the cobrapy model, meaning that there will be a ``_e0`` or ``_e`` appended to the names. We plan to correct this in future versions so that MetConSIN will recognize metabolites without the appended ``_e0`` or ``_e``. 

.. warning::

    All models in the community must use the same exchanged metabolite tag (e.g. ``_e0``).



Setting metabolic uptake rate parameters
---------------------------------------------

Dynamic FBA requires some mapping from the environmental metabolites to a set of bounds on the exchange reaction. In this tutorial, we assume that lower bounds are constant, and upper bounds are simply linear in the amount of metabolite available. By defualt, 
MetConSIN will assume the constants of parameters of these linear functions are uniformly 1. However, if we'd like to load in some parameters that we have perhaps fit to data. The parameters should
be passed as a dictionary keyed by the model names. Each entry in that dictionary can either be an array, ordered according to the model's ordering of the metabolites (which we probably don't want to try to figure out) or, more conveniently, a dictionary keyed by metabolite
names. Python dictionaries can be easily saved and loaded using the ``.json`` file format. For example:

.. code-block:: python

    with open("exchange_bounds_uniform.json") as fl:
        uptake_params = json.load(fl)

loads a set of parameters that are all uniformly 1. For parameters chosen at random from the interval :math:`[0.5:1.5]`, you can instead do

.. code-block:: python

    with open("exchange_bounds_made_up.json") as fl:
        uptake_params = json.load(fl)

Currently, MetConSIN supports constant bounds, linear bounds, or Hill function bounds by keyword, as well as allowing user defined bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>` for details on how to use other bounds.


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


To run MetConSIN, we call :py:func:`metconsin_sim <metconsin.metconsin_sim>`, passing our growth media, how long we'd like the simulation to run for, as well as a choice of metabolic uptake bound functions.

.. code-block:: python

    initial_abundance = dict([(sp,0.1) for sp in species])

    with open("example.log",'w') as fl:
        metconsin_return = metconsin_sim(species,model_info_fl,initial_abundance = initial_abundance,endtime = 2,media = growth_media, ub_funs = "linear",ub_params = uptake_params,flobj = fl,resolution = 0.01)

We set the intial abundance of each microbe using a dictionary keyed by the microbe names.

By default, MetConSIN prints a log of its activity. Here, we redirect this log to the file ``example.log`` by passing the file with the ``flobj`` parameter.

The results can be saved using the :py:func:`save_metconsin <metconsin.save_metconsin>` function:

.. code-block:: python

    flder = os.path.join(flder,"metconsin_results")

    save_metconsin(metconsin_return, flder)

:py:func:`save_metconsin <metconsin.save_metconsin>` saves the simulation dynamics in two tab-separated files: ``Microbes.tsv`` and ``Metabolites.tsv`` with rows corresponding to state variables (microbes or metabolites) and columns
corresponding to time-points. It also creates plots of the simulation dynamics (although these are not publication quality) and saves a list of times that the bases were changed for any microbe (as a table of bools indexed by model with columns basis change times.)

Finally, it creates a set of sub-directories to save internal and exchange fluxes, as well as the sequence of interaction networks.

Improved Plotting
--------------------

While :py:func:`save_metconsin <metconsin.save_metconsin>` plots the simulation, it may not produce the nicest looking plots. Because we have only 10 species in our simulation,
we can use a 10-color set (matplotlib's ``tab10`` colormap) to color-code the vertical lines we use to indicate basis changes:

.. code-block:: python

    fig,ax = plt.subplots(figsize = (30,10))
    metconsin_return["Microbes"].T.plot(ax = ax,colormap = "tab10")
    ax.set_xlim(0,4)
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,50)
    cx = np.arange(0,1,0.1)
    cmap = plt.cm.tab10.colors
    cdict = dict([(metconsin_return["Microbes"].index[i],cmap[i]) for i in range(10)])
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
        else:
            col = cdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,"o",color = col)

Furthermore, the ``Metabolite.png`` plot produced by :py:func:`save_metconsin <metconsin.save_metconsin>` plots all of environmental metabolites, which is too many for a 
useful figure. Instead, let's only plot the metabolites that are produced:

.. code-block:: python

    fig,ax = plt.subplots(figsize = (30,10))
    f = lambda x: np.any(x>x[0])
    produced = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f,axis = 1)]
    produced.T.plot(ax = ax,colormap = "tab20")#,legend = False)
    ax.set_xlim(0,4)
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,50)
    cx = np.arange(0,1,0.1)
    cmap = plt.cm.tab10.colors
    cdict = dict([(metconsin_return["Microbes"].index[i],cmap[i]) for i in range(10)])
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
        else:
            col = cdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,"o",color = col)
    plt.savefig("produced_metabolites.png")


.. code-block:: python

    fig,ax = plt.subplots(figsize = (30,10))
    f = lambda x: np.any(x<0.8*x[0])
    consumed = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f,axis = 1)]
    consumed.T.plot(ax = ax,colormap = "tab20")#,legend = False)
    ax.set_xlim(0,4)
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,50)
    cx = np.arange(0,1,0.1)
    cmap = plt.cm.tab10.colors
    cdict = dict([(metconsin_return["Microbes"].index[i],cmap[i]) for i in range(10)])
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
        else:
            col = cdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,"o",color = col)
    plt.savefig("consumed_metabolites.png")


Analyzing the networks
---------------------------

To demonstrate the value of MetConSIN, we include some network analysis of the networks we created.

The Species-Metabolite networks
+++++++++++++++++++++++++++++++++

The specie-metabolite networks are bipartite networks of microbes and metabolites. In this tutorial, we explore the network connectivity of the microbe nodes using 
a couple of helper functions - :py:func:`make_microbe_table <analysis_helpers.make_microbe_table>` and :py:func:`make_microbe_growthlimiter <analysis_helpers.make_microbe_growthlimiter>`.

These functions identify the metabolites that have a direct effect on microbial growth (the rate-limiting metabolites) in each time range. The following code creates tables of 
rate limiting-metabolites for each microbe in our community, and plots the coefficients for those rate-limiting metabolites in the growth equation of the microbe.

.. code-block:: python

    for mic in species:
        microbe_results = ah.make_microbe_table(mic,metconsin_return["SpcMetNetworks"])
        microbe_results.to_csv("{}_networkinfo.tsv".format(mic),sep = '\t')
        grth_cos = ah.make_microbe_growthlimiter(mic,metconsin_return["SpcMetNetworks"])
        fig,ax = plt.subplots(figsize = (20,10))
        sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Metabolite",ax=ax)
        ax.set_title("{} Limiting Metabolites".format(mic))
        plt.savefig("{}_limiting_metabolites.png".format(mic))

The next block of code finds the set of metabolites which appear as rate limiting for any microbe in any time-range. It then makes a table for each limiting metabolite of coefficients in the growth
equation of each microbe at each time range, and plots the result.

.. code-block:: python

    all_limiters = []
    for ky in metconsin_return["SpcMetNetworks"].keys():
        df = metconsin_return["SpcMetNetworks"][ky]['edges']
        all_limiters += list(df[df["SourceType"] == "Metabolite"]["Source"])
    all_limiters = np.unique(all_limiters)

    for limi in all_limiters:
        limtab = ah.make_limiter_table(limi,metconsin_return["SpcMetNetworks"],species)
        limtab.to_csv("{}_limiter.csv".format(limi),sep = '\t')
        fig,ax = plt.subplots(figsize = (20,10))
        grth_cos = ah.make_limiter_plot(limi,metconsin_return["SpcMetNetworks"])
        sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Model",ax=ax)
        ax.legend(loc=2)
        ax.set_title("{} As Growth Limiter".format(limi))
        plt.savefig("{}_limiter_plot.png".format(limi))

Metabolite-Metabolite networks
+++++++++++++++++++++++++++++++++++

The last analysis we will present is of the metabolite-metabolite networks. Here, we have a weighted, directed network suitable for many network analysis algorithms. Additionally,
there is a set of such networks. We will inspect how these networks change across the time-intervals of simulation by looking for the edges with the highest variance in weight, as well
as the nodes (i.e. metabolites) with the highest variance in degree.

The highest variance edges can be found by sorting the average network.

.. code-block:: python

    metconsin_return["MetMetNetworks"]['Combined']['edges'].sort_values("Variance",ascending=False).head(10).to_latex(os.path.join(flder,"MetMetHighestVarEdges.tex"))

The last block of code uses :py:func:`node_in_stat_distribution <analysis_helpers.node_in_stat_distribution>` and :py:func:`node_out_stat_distribution <analysis_helpers.node_out_stat_distribution>`
to create tables that summarize the degrees of the nodes across the networks (in and out seperately). We find the average and the variance of the following for each node

- Number of edges connected to the node
- Sum of the weights of those edges
- Sum of the absolute value of the weights of those edges
- Sum of the weights of the positive weighted edges connected to the node
- Sum of the absolute value of the weights of the negative weighted edges connected to the node

We then sort by highest variance total weight.

.. code-block:: python

    ### The network making cleans up the names.
    metabolite_list = [met.replace("_e0","").replace("_e","") for met in np.array(metconsin_return["Metabolites"].index)]

    avg_in_degrees, var_in_degrees, in_zeros = ah.node_in_stat_distribution(metabolite_list,metconsin_return["MetMetNetworks"])
    avg_out_degrees, var_out_degrees, in_zeros = ah.node_out_stat_distribution(metabolite_list,metconsin_return["MetMetNetworks"])

    avg_in_degrees.to_csv(os.path.join(flder,"MetMetNodeInAvg.tsv",sep = '\t'))
    var_in_degrees.to_csv(os.path.join(flder,"MetMetNodeInVar.tsv",sep = '\t'))

    avg_out_degrees.to_csv(os.path.join(flder,"MetMetNodeOutAvg.tsv",sep = '\t'))
    var_out_degrees.to_csv(os.path.join(flder,"MetMetNodeOutVar.tsv",sep = '\t'))

    highest_in_var = var_in_degrees.sort_values("SumWeight",ascending = False).head(10)
    highest_in_var.to_latex(os.path.join(flder,"highest_node_in_variance.tex"))
    avg_in_degrees.loc[highest_in_var.index].to_latex(os.path.join(flder,"highest_node_in_var_average.tex"))

    highest_out_var = var_out_degrees.sort_values("SumWeight",ascending = False).head(10)
    highest_out_var.to_latex(os.path.join(flder,"highest_node_out_variance.tex"))
    avg_out_degrees.loc[highest_out_var.index].to_latex(os.path.join(flder,"highest_node_out_var_average.tex"))