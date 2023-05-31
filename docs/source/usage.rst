Usage
=====

.. _installation:

Installation
------------

To use MetConSIN, clone from github:

.. code-block:: console

    $ git clone https://github.com/jdbrunner/metconsin.git

(We plan to add pip installation in the future)

Because we haven't implemented pip installation yet, you will need to add the directory to your python path. This can be done within the script using

.. code-block:: python

    import sys
    import os
    sys.path.append(os.path.join(os.path.expanduser("~"),location,"metconsin"))

where ``location`` is the path to the folder that you cloned MetConSIN into. Alternatively, make sure the ``metconsin`` directory is in your ``$PATH`` variable.

**Dependencies**

MetConSIN requires `Gurobi <https://www.gurobi.com/documentation/9.5/>`_ and gurobi's python package for linear the programming (basis finding) steps. Alternatively, MetConSIN can use the open source `CyLP <http://mpy.github.io/CyLPdoc/index.html>`_, but this is much slower.

For COBRA models, MetConSIN uses the `cobrapy <https://opencobra.github.io/cobrapy/>`_ package.

Metconsin also uses `numba <https://numba.pydata.org/>`_ to speed up computation.

.. note:: 

    We do not currently support CPLEX, because we do not have access to a CPLEX license. Earlier versions of SurfinFBA supported CPLEX, but these versions are now depreciated. And not available in this package.


Generating Simulations and Networks
-------------------------------------

The basic usage of MetConSIN is to generate a dynamic FBA simulation and the corresponding sequence of networks using the :py:func:`metconsin_sim <metconsin.metconsin.metconsin_sim>` function

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

The networks are contained in dictionaries keyed by "edges" and "nodes", corresponding to a list of edges and list of nodes as pandas dataframes. The :py:func:`save_metconsin <metconsin.metconsin.save_metconsin>` function will save the results in a directory. Networks and simulated dynamics will be saved as .tsv files and plots will be made of the dynamics.

Alternatively, one can do the following:

.. code-block:: python

    from metconsin import metconsin_network
    metconsin_return = metconsin_network(community_members,model_info_file, save_folder,**kwargs)

to use the :py:func:`metconsin_network <metconsin.metconsin_network>` function to construct and save a single set of networks corresponding to the initial media. Note that these networks may not be stable.

Our previous dynamic FBA simulator, surfinFBA is included in this project. To compute a dynamic FBA simulation (without generating the corresponding networks), use :py:func:`dynamic_fba <metconsin.metconsin.dynamic_fba>`. Its usage is similar to :py:func:`metconsin_sim <metconsin.metconsin.metconsin_sim>`, but it returns (and optionally saves to .tsv files) just the dynamics of the simulation. 
For example, to simulate with dFBA and store the microbial and metabolic dynamics in pandas dataframes, simply run:

.. code-block:: python
    
    from metconsin import dynamic_fba
    dynamics = dynamic_fba(community_members,model_info_file)
    Microbes = dynamics["Microbes"]
    Metabolites = dynamics["Metabolites"]

Using AGORA and user-defined media
-------------------------------------

We include a function in ``AGORA_Media`` to download the full set of media files avaialble at `AGORA <https://www.vmh.life/>`_. Simply run

.. code-block:: python

    from metconsin import dynamic_fba
    get_AGORA_diets()


will download all avaialble media as well as metadata. 

After download, :py:func:`metconsin_environment<metconsin.metconsin.metconsin_environment>` can be used to generate an initial environment for a set of genome-scale models from an AGORA file, or
another table. For example, to create an environment from a file located at ``path/to/media.tsv`` for a set of models generated by modelSEED, we can do

.. code-block:: python

    environment = metconsin_environment(community_members,model_info_file,media_source='path/to/media.tsv', metabolite_id_type = 'modelSeedID')

The media file should have a column labeled ``modelSeedID`` with the modelSEED ID for each metabolite, and a column labled ``fluxValue`` for the media dictionary value. 

.. note::

    Diet files often do not contain all of the metabolites that can be exchanged by the models, particularly if the diet file and model come from different sources (e.g. an AGORA diet file and a modelSEED model).
    Use the parameter ``default_proportion`` to set the availability (as a proportion of the model medium flux) for any metabolite not included in the diet file. Setting ``default_proportion`` to 0 will result in an environment
    set exactly by the provided table, but may result in 0 growth.

For convenience, we do not need the full path to the diet files cretaed by ``download_media.py``, provided that they are present in the ``AGORA_Media`` folder. Instead, we can use AGORA's names. For example:

.. code-block:: python

    environment = metconsin_environment(community_members,model_info_file,media_source='EU average', metabolite_id_type = 'modelSeedID')

will create an environment from the ``EU average`` diet.

Finally, we can create a minimal media (actually the maximum of the minimal medias for each model), including with a specified initial growth rate for each model:

.. code-block:: python

    environment = metconsin_environment(community_members,model_info_file,media_source='minimal',intitial_growth = 10)