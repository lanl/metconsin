Usage
=====

.. _installation:

Installation
------------

To use MetConSIN, clone from github:

.. code-block:: console

    $ git clone https://github.com/jdbrunner/metconsin.git

(We plan to add pip installation in the future)

You will also need to add the directory to your python path, for example using

.. code-block:: python

    import sys
    import os
    sys.path.append(os.path.join(os.path.expanduser("~"),location,"metconsin"))

where ``location`` is the path to the folder that you cloned MetConSIN into.

**Dependencies**

MetConSIN requires `Gurobi <https://www.gurobi.com/documentation/9.5/>`_ and gurobi's python package for linear the programming (basis finding) steps. Alternatively, MetConSIN can use the open source `CyLP <http://mpy.github.io/CyLPdoc/index.html>`_, but this is much slower.

Metconsin also uses `numba <https://numba.pydata.org/>`_ to speed up computation.


Generating Simulations and Networks
-------------------------------------

The basic usage of MetConSIN is to generate a dynamic FBA simulation and the corresponding sequence of networks using the :py:func:`metconsin_sim <metconsin.metconsin_sim>` function

.. code-block:: python

    from metconsin import metconsin_sim,save_metconsin
    metconsin_return = metconsin_sim(community_members,model_info_file,**kwargs)
    save_metconsin(metconsin_return,"results")

where ``community_members`` is a list of the taxa in the community being modeled and ``model_info_file`` is the path to a .csv file indicating the paths to the GEMs for each community member. The keyword arguments can be used to change options including simulation length, metabolite inflow/outflow, simulation resolution, etc.

``metconsin_return`` will be a dictionary with the following items:

    - *Microbes*\ : dynamics of the microbial taxa, as a pandas dataframe
    - *Metabolites*\ : dynamics of the metabolites, as a pandas dataframe
    - *SpeciesNetwork*\ : Species-Species networks defined by simple hueristic (keyed by time interval)
    - *MetMetNetworks*\ : Metabolite-Metabolite networks defined by the dfba sequence of ODEs (keyed by time interval)
    - *SpcMetNetworkSummaries*\ : Microbe-Metabolite networks defined by the dfba sequence of ODEs with all edges between two nodes collapsed to one edge (keyed by time interval)
    - *SpcMetNetworks*\ : Microbe-Metabolite networks defined by the dfba sequence of ODEs (keyed by time interval)
    - *BasisChanges*\ : Times that the system updated a basis
    - *ExchangeFluxes*\ (if ``track_fluxes`` == True): Exchange fluxes at each time-point for each taxa.
    - *InternalFluxes*\ (in ``save_internal_flux`` == True): Internal fluxes at each time-point for each taxa.

The networks are contained in dictionaries keyed by "edges" and "nodes", corresponding to a list of edges and list of nodes as pandas dataframes. The :py:func:`save_metconsin <metconsin.save_metconsin>` function will save the results in a directory. Networks and simulated dynamics will be saved as .tsv files and plots will be made of the dynamics.

Alternatively, one can do the following:

.. code-block:: python

    from metconsin import metconsin_network
    metconsin_return = metconsin_network(community_members,model_info_file, save_folder,**kwargs)

to use the :py:func:`metconsin_network <metconsin.metconsin_network>` function to construct and save a single set of networks corresponding to the initial media. Note that these networks may not be stable.

Our previous dynamic FBA simulator, surfinFBA is included in this project. To compute a dynamic FBA simulation (without generating the corresponding networks), use :py:func:`dynamic_fba <metconsin.dynamic_fba>`. Its usage is similar to :py:func:`metconsin_sim <metconsin.metconsin_sim>`, but it returns (and optionally saves to .tsv files) just the dynamics of the simulation. 
For example, to simulate with dFBA and store the microbial and metabolic dynamics in pandas dataframes, simply run:

.. code-block:: python
    
    from metconsin import dynamic_fba
    dynamics = dynamic_fba(community_members,model_info_file)
    Microbes = dynamics["Microbes"]
    Metabolites = dynamics["Metabolites"]