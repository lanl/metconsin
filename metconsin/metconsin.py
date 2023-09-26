from . import make_network as mn
# import surfmod as sm
from . import prep_models as pr
from . import dynamic_simulation as surf
import time
import pandas as pd
import cobra as cb
import numpy as np
import contextlib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from functools import singledispatch

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def metconsin_environment(community_members,model_info_file,**kwargs):
    """
    Generates an intitial environment dictionary suitable for loading into metconsin with the community members to be simulated.

    :param community_members: list of models of taxa in the community of interest
    :type community_members: list[str]
    :param model_info_file: path to csv with GEM info. csv file should contain a column called "Species" with names that include the names in ``desired_models``, and a column called "File" that contains the path to a GSM in xml or json form (note that .mat is not currently supported).
    :type model_info_file: str

    :param media_source: Source of desired environment. Can be a path to a file with a media table, a choice of media from the AGORA list (see below), ``minimal`` for an environment created from minimal media, or None for uniform availability of metabolites. Default None.
    :type media_source: str

    :param metabolite_id_type: Column heading of ``media_source`` that labels the metabolites in such a way as to mactch the labeling in the models. **We assume this is the same for all models**. Default "metabolite"
    :type metabolite_id_type: str

    :param default_proportion: For exchanged metabolites not found in the supplied media, the environment will contain concentration of the metbaolite equal to the max flux value in the model medias across models times this parameter. Default 0.1
    :type default_proportion: float
    :param intitial_growth: Option to constrain initial growth when computing minimal media. If None, uses model growth from model default media. Default None.
    :type intitial_growth: float

    .. note::

        The available media in the AGORA folder (once :py:func:`get_AGORA_diets` has been run) can be seen at the `AGORA website <https://www.vmh.life/#nutrition>`_. Rerun :py:func:`get_AGORA_diets` for up to date media.

    :return: dictionary describing initial environment for community
    :rtype: dict
    """

    media_file = kwargs.get("media_source")
    met_id_col = kwargs.get("metabolite_id_type","metabolite")
    default_proportion = kwargs.get("default_proportion",0.1)
    minimal_grth = kwargs.get("intitial_growth")

    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in community_members:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            if flnm.split(".")[-1] == "json":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.load_json_model(flnm)
            elif flnm.split(".")[-1] == "xml":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.read_sbml_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)

    if media_file == "minimal":

        print("[metconsin_environment] Creating environment from minimal media.")
        growth_media = pr.make_media(cobra_models,default_proportion = 1,minimal=True,minimal_grth=minimal_grth).to_dict()

    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    agora_flder = "AGORA_Media"
    agora_media_loc = os.path.join(parent,agora_flder)

    if isinstance(media_file,str):

        if "{}_AGORA.tsv".format(media_file) in os.listdir(agora_media_loc):
            media_file = os.path.join(agora_media_loc,"{}_AGORA.tsv".format(media_file))

        if "{}_AGORA.tsv".format(media_file.replace(" ","_")) in os.listdir(agora_media_loc):
            media_file = os.path.join(agora_media_loc,"{}_AGORA.tsv".format(media_file.replace(" ","_")))

        if media_file in os.listdir(agora_media_loc):
            media_file = os.path.join(agora_media_loc,media_file)

        if media_file == "minimal":
            print("[metconsin_environment] Creating environment from minimal media.")
            growth_media = pr.make_media(cobra_models,default_proportion = 1,minimal=True,minimal_grth=minimal_grth).to_dict()
        elif os.path.isfile(media_file):
            print("[metconsin_environment] Creating environment from file {}".format(media_file))
            growth_media = pr.make_media(cobra_models,media_df = media_file,metabolite_id_type=met_id_col,default_proportion = default_proportion).to_dict()
        else:
            print("[metconsin_environment] No file {}. Creating default environment from model mediums".format(media_file))
            growth_media = pr.make_media(cobra_models,default_proportion =default_proportion).to_dict()

    elif media_file == None:
        print("[metconsin_environment] Creating default environment from model mediums")
        growth_media = pr.make_media(cobra_models,default_proportion =default_proportion).to_dict()

    else:
        print("[metconsin_environment] Parameter media_file should be a str or None. Creating default environment from model mediums")
        growth_media = pr.make_media(cobra_models,default_proportion =default_proportion).to_dict()

    return growth_media



def metconsin_network(community_members,model_info_file, save_folder,**kwargs):

    '''


    Creates dFBA implied network (microbe-metabolite, metabolite-metabolite, heuristic microbe-microbe) for community interaction from GEMs with media that reflect the desired environment.
    If no media is provided, the default "diet" files for the given COBRA models is averaged. Saves the results as .tsv files


    :param community_members: list of models of taxa in the community of interest
    :type community_members: list[str]
    :param model_info_file: path to csv with GEM info. csv file should contain a column called "Species" with names that include the names in ``desired_models``, and a column called "File" that contains the path to a GSM in xml or json form (note that .mat is not currently supported).
    :type model_info_file: str
    :param save_folder: desired path to save network output
    :type save_folder: str
    :param media: Growth media used for the simulation. If None, averages the media dicts packaged with the GSMs used. Can be keyed by metabolite name, ID, or exchange reaction ID. Alternatively, passing the string "minimal" attempts to define a minimal growth media for each microbe and averages that. Default None.
    :type media: dict or str
    :param metabolite_inflow: Inflow rate for each metabolite. Default all 0
    :type metabolite_inflow: array[float]
    :param metabolite_outflow: Outflow rate for each metabolite. Default all 0
    :type metabolite_outflow: array[float]

    
    :param ub_funs: General function to use for upper bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``.  Default ``linear``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type ub_funs: str
    :param lb_funs: General function to use for lower bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``. Default ``constant``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type lb_funs: str
    :param ub_params: Parameters to use for upper bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type ub_params: dict
    :param lb_params: Parameters to use for lower bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type lb_params: dict

    :param upper_bound_user_functions: User-defined upper bound functions for exchange reactions. **Ignored** unless ``ub_funs`` == ``user``. Any model not included (which can be all models), will default to functions defined by the ``ub_funs`` parameter. Default None.
    :type upper_bound_user_functions: dict[str,array of lambdas]
    :param lower_bound_user_functions: User-defined lower bound functions for exchange reactions. **Ignored** unless ``lb_funs`` == ``user``.Any model not included (which can be all models), will default to functions defined by the ``lb_funs`` parameter. Default None.
    :type lower_bound_user_functions: dict[str,array of lambdas]    
    :param upper_bound_user_functions_dt: Derivatives of user defined upper_bound_functions_dt. Default None
    :type upper_bound_user_functions_dt:  dict[str,array of lambdas]
    :param lower_bound_user_functions_dt: kwargs.get(lower_bound_functions_dt)
    :type lower_bound_user_functions_dt:  dict[str,array of lambdas]
    
    
    :param solver: LP solver to use (currently supports ``gurobi`` and ``clp``). Default ``gurobi``
    :type solver: str
    :param met_filter: If ``met_filter_sense`` == "exclude", list of metabolites to treat as infinitely supplied (i.e. ignored in the dynamics). If ``met_filter_sense`` == "include", all other metabolites will be treated as infinitely supplied (i.e. ignored in the dynamics). Default None
    :type met_filter: list
    :param met_filter_sense: Choice of "include" or "exclude" for metabolite filter. If left as None, no filter will be done. Default None
    :type met_filter: str
    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File
    :param report_activity: Whether or not to log simulation progress. Default True
    :type report_activity: bool
    :param report_activity_network: Whether or not to log network building progress. Default True
    :type report_activity_network: bool
    :param forceOns: Whether or not to allow internal reactions to be forced on (with a positive lower bound). Many GSM include such bounds. Default True
    :type forceOns: bool
    :param secondobj: secondary objective for FBA linear program. Can be "total" to minimize total flux at optimum, or array representing linear objective. Default total
    :type secondobj: str or numpy array
    :param trimit: Whether or not to remove metabolites from the networks if they are not present in the media. Default True
    :type trimit: bool
    :param returnNets: Whether or not to return the networks as a dictionary. Otherwise, they are only saved to .tsv files. Default False
    :type returnNets: bool

    
    .. note::

        When setting bound functions and parameters, any models left out of a parameter dictionary will use default parameters for the choice of function type. Currently, the only way to use different function **types** for each model is to use user defined functions. 
        
    .. warning::

        When creating user defined functions, pay attention to issues with variable scope! See `python docs <https://docs.python.org/3.4/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result>`_ for more information.


    
    :return: Dictionary containing the simulation and networks. Keys are :

    - *Resource Mediated Nodes*\ : Node data for microbe-metabolite network
    - *Resource Mediated Edges (Full)*\ : All edges for microbe-metabolite network
    - *Resource Mediated Edges (Summary)*\ : Edges of microbe-metabolite network with all edges between two nodes collapsed to one edge
    - *Metabolite Edges*\ : Edges of metabolite-metabolite network
    - *Metabolite Nodes*\ : Node data for metabolite-metabolite network
    - *Species Edges*\ : Edges of species-species network defined by simple hueristic
    - *Species Nodes*\ : Node data for species-species network defined by simple hueristic
    - *Species Adjacency*\ : Adjacency matrix of species-species network defined by simple hueristic

    :rtype: dict[pandas dataframes]

    '''


    metabolite_inflow = kwargs.get("metabolite_inflow")
    metabolite_outflow = kwargs.get("metabolite_outflow")
    solver = kwargs.get("solver",'gurobi')
    flobj = kwargs.get("flobj")
    media = kwargs.get("media")
    report_activity = kwargs.get("report_activity",True)
    report_activity_network= kwargs.get("report_activity_network",True)
    secondobj = kwargs.get("secondobj","total")
    trimit = kwargs.get("trimit",True)
    returnNets = kwargs.get("returnNets",False)




    start_time = time.time()

    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in community_members:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            if flnm.split(".")[-1] == "json":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.load_json_model(flnm)
            elif flnm.split(".")[-1] == "xml":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.read_sbml_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)


    print("Loaded " + str(len(cobra_models)) + " models successfully")

    if media == "minimal":
        for model in cobra_models.keys():
            mxg = cobra_models[model].slim_optimize()
            try:
                min_med = cb.medium.minimal_medium(cobra_models[model],mxg,minimize_components = True)
                cobra_models[model].medium = min_med
            except:
                pass
        kwargs["media"] = {}


    for mod in cobra_models.keys():
        print(mod," COBRA initial growth rate: ",cobra_models[mod].slim_optimize())

    #returns dict of surfmods, list of metabolites, and concentration of metabolites.
    # models,mets,mets0 = prep_cobrapy_models(cobra_models,uptake_dicts = uptake_dicts ,random_kappas=random_kappas)


    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,**kwargs)

    y0 = np.array([y0dict[met] for met in metlist])
    ydot0 = np.zeros_like(y0)

    try:
        if len(metabolite_inflow) != len(y0):
            metabolite_inflow = np.zeros_like(y0)
    except:
        if metabolite_inflow == None:
            metabolite_inflow = np.zeros_like(y0)
    try:
        if len(metabolite_outflow) != len(y0):
            metabolite_outflow = np.zeros_like(y0)
    except:
        if metabolite_outflow == None:
            metabolite_outflow = np.zeros_like(y0)

    for ky,model in models.items():


        model.ezero  = 10**-8

        print("Testing Initial: {}".format(ky))

        metabolite_con = y0[model.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

        if solver == 'gurobi':
            obval = model.fba_gb(y0,secondobj = secondobj,report_activity = report_activity,flobj = flobj)
        elif solver == 'clp':
            obval = model.fba_clp(y0,secondobj = secondobj,report_activity = report_activity,flobj = flobj)
        else:
            print("Invalid Solver. Choose 'gurobi' or 'clp'")
    ###

        ydi = np.zeros_like(ydot0)
        ydi[model.ExchangeOrder] = -np.dot(model.GammaStar,model.inter_flux)
        ydot0 += ydi


    for ky,model in models.items():
        
        print("Findwaves: {}".format(ky))

        model.findWave(y0,ydot0,details = report_activity,flobj = flobj)

        model.compute_internal_flux(y0)


    try:
      os.mkdir(save_folder)
    except:
      pass
    rmnodes,rmfull,rmsumm,mmedges,mmnodes = mn.species_metabolite_network(metlist,y0,models,report_activity=report_activity_network,flobj=flobj)

    if trimit:

        modellist = [model for model in models.values()]
        dynamics = pd.DataFrame(index = [mod.Name for mod in modellist]+list(metlist))
        s0 = np.concatenate([np.ones(len(modellist)),y0])
        dynamics_vals = surf.evolve_community(0,s0,modellist,metabolite_inflow,metabolite_outflow)
        dynamics["s0"] = s0
        dynamics["sdot0"] = dynamics_vals

        mmedges,mmnodes = mn.trim_network(mmedges,mmnodes,dynamics)
        rmfull,_ = mn.trim_network(rmfull,rmnodes,dynamics)
        rmsumm,rmnodes = mn.trim_network(rmsumm,rmnodes,dynamics)

    ssnet,ssnodes,ssadj=mn.heuristic_ss(rmsumm,rmnodes,report_activity=report_activity_network)

    rmsumm
    ssnet

    rmnodes.to_csv(os.path.join(save_folder,"microbe_metabolite_nodes.tsv"),sep = '\t')
    rmfull.to_csv(os.path.join(save_folder,"microbe_metabolite_full_edges.tsv"),sep = '\t')
    rmsumm.to_csv(os.path.join(save_folder,"microbe_metabolite_summary_edges.tsv"),sep = '\t')
    mmedges.to_csv(os.path.join(save_folder,"metabolite_metabolite_edges.tsv"),sep = '\t')
    mmnodes.to_csv(os.path.join(save_folder,"metabolite_metabolite_nodes.tsv"),sep = '\t')
    ssnet.to_csv(os.path.join(save_folder,"microbe_microbe_edges.tsv"),sep = '\t')
    ssnodes.to_csv(os.path.join(save_folder,"microbe_microbe_nodes.tsv"),sep = '\t')
    ssadj.to_csv(os.path.join(save_folder,"microbe_microbe_adjacency.tsv"),sep = '\t')


    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Networks built in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN]: Networks built in ",int(minuts)," minutes, ",sec," seconds.")
    
    if returnNets:
        return {"Resource Mediated Nodes":rmnodes,"Resource Mediated Edges (Full)":rmfull,"Resource Mediated Edges (Summary)":rmsumm
                ,"Metabolite Edges":mmedges,"Metabolite Nodes":mmnodes,"Species Edges":ssnet,"Species Nodes":ssnodes,"Species Adjacency":ssadj}
    else:
        return True



def metconsin_sim(community_members,model_info_file,**kwargs):

    '''

    Does Full Simulation, makes networks for each time interval.

    :param community_members: list of models of taxa in the community of interest
    :type community_members: list[str]
    :param model_info_file: path to csv with GEM info. csv file should contain a column called "Species" with names that include the names in ``desired_models``, and a column called "File" that contains the path to a GSM in xml or json form (note that .mat is not currently supported).
    :type model_info_file: str

    :param initial_abundance: intial abundances (treated as absolute) of the community to be simulated. Should be a dictionary keyed by community members. Default uniform community
    :type initial_abundance: dict[str,float]

    :param endtime: Simulation length. Default 10**-2
    :type endtime: float
    :param resolution: Time-resolution of the dynamics output. Default 0.1
    :type resolution: float
    :param media: Growth media used for the simulation. If None, averages the media dicts packaged with the GSMs used. Can be keyed by metabolite name, ID, or exchange reaction ID. Alternatively, passing the string "minimal" attempts to define a minimal growth media for each microbe and averages that. Default None.
    :type media: dict or str
    :param metabolite_inflow: Inflow rate for each metabolite. Dictionary keyed like media. Default all 0
    :type metabolite_inflow: dict
    :param metabolite_outflow: Outflow rate for each metabolite. Dictionary keyed like media. Default all 0
    :type metabolite_outflow: dict
    :param model_deathrates: Decay rate for each community member, given as dictionary keyed by community member names. Defaul all 0
    :type model_deathrates: dict[str,float]


    :param ub_funs: General function to use for upper bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``.  Default ``linear``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type ub_funs: str
    :param lb_funs: General function to use for lower bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``. Default ``constant``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type lb_funs: str
    :param ub_params: Parameters to use for upper bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type ub_params: dict
    :param lb_params: Parameters to use for lower bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type lb_params: dict

    :param upper_bound_user_functions: User-defined upper bound functions for exchange reactions. **Ignored** unless ``ub_funs`` == ``user``. Any model not included (which can be all models), will default to functions defined by the ``ub_funs`` parameter. Default None.
    :type upper_bound_user_functions: dict[str,array of lambdas]
    :param lower_bound_user_functions: User-defined lower bound functions for exchange reactions. **Ignored** unless ``lb_funs`` == ``user``.Any model not included (which can be all models), will default to functions defined by the ``lb_funs`` parameter. Default None.
    :type lower_bound_user_functions: dict[str,array of lambdas]    
    :param upper_bound_user_functions_dt: Derivatives of user defined upper_bound_functions_dt. Default None
    :type upper_bound_user_functions_dt:  dict[str,array of lambdas]
    :param lower_bound_user_functions_dt: kwargs.get(lower_bound_functions_dt)
    :type lower_bound_user_functions_dt:  dict[str,array of lambdas]


    :param solver: LP solver to use (currently supports ``gurobi`` and ``clp``). Default ``gurobi``
    :type solver: str
    :param met_filter: If ``met_filter_sense`` == "exclude", list of metabolites to treat as infinitely supplied (i.e. ignored in the dynamics). If ``met_filter_sense`` == "include", all other metabolites will be treated as infinitely supplied (i.e. ignored in the dynamics). Default None
    :type met_filter: list
    :param met_filter_sense: Choice of "include" or "exclude" for metabolite filter. If left as None, no filter will be done. Default None
    :type met_filter: str
    :param track_fluxes: Whether or not to save the exchange fluxes computed during the simulation. Default True
    :type track_fluxes: bool
    :param save_internal_flux: Whether or not to save the internal fluxes computed during the simulation. Default True
    :type save_internal_flux: bool
    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File
    :param final_interval_weight: Weight to give final network in time averaging. Default = 0.1
    :type final_interval_weight: float
    :param report_activity: Whether or not to log simulation progress. Default True
    :type report_activity: bool
    :param report_activity_network: Whether or not to log network building progress. Default True
    :type report_activity_network: bool
    :param findwaves_report: Whether or not to log details of basis finding. Default False
    :type findwaves_report: bool
    :param debugging: Turn on some debugging prints. Default False
    :type debugging: bool
    :param forceOns: Whether or not to allow internal reactions to be forced on (with a positive lower bound). Many GSM include such bounds. Default True
    :type forceOns: bool
    :param refine_intervals: whether to look for interval endtimes earlier than provided by solve_ivp "events". Default False
    :type refine_intervals: bool

    .. note::

        When setting bound functions and parameters, any models left out of a parameter dictionary will use default parameters for the choice of function type. Currently, the only way to use different function **types** for each model is to use user defined functions. 
        
    .. warning::

        When creating user defined functions, pay attention to issues with variable scope! See `python docs <https://docs.python.org/3.4/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result>`_ for more information.


    :return: Dictionary containing the simulation and networks. Keys are :

    - *Microbes*\ : dynamics of the microbial taxa, as a pandas dataframe
    - *Metabolites*\ : dynamics of the metabolites, as a pandas dataframe
    - *SpeciesNetwork*\ : Species-Species networks defined by simple hueristic (keyed by time interval)
    - *MetMetNetworks*\ : Metabolite-Metabolite networks defined by the dfba sequence of ODEs (keyed by time interval)
    - *SpcMetNetworkSummaries*\ : Microbe-Metabolite networks defined by the dfba sequence of ODEs with all edges between two nodes collapsed to one edge (keyed by time interval)
    - *SpcMetNetworks*\ : Microbe-Metabolite networks defined by the dfba sequence of ODEs (keyed by time interval)
    - *BasisChanges*\ : Times that the system updated a basis with bools indicating if a particular model updated at that time.
    - *InternalBasisDifferences*\ : List of constraints/variables that are activated/deactivated at each basis change for each model.
    - *BasisChangeCause*\ : List of bounds that were violated leading a basis change.
    - *ExchangeFluxes*\ (if ``track_fluxes`` == True): Exchange fluxes at each time-point for each taxa.
    - *InternalFluxes*\ (in ``save_internal_flux`` == True): Internal fluxes at each time-point for each taxa.

    :rtype: dict 

    .. note::

        Networks include ``Average``, ``Combined``, and ``Difference`` networks along with individual networks. ``Average`` is time-weighted average, ``Combined`` includes edge weights for each time interval in a single table, and ``Difference`` is the change in edge weights 
        at a basis change, Old - New.

    '''

    

    final_interval_weight = kwargs.get("final_interval_weight",0.1)
    metabolite_inflow = kwargs.get("metabolite_inflow",{})
    metabolite_outflow = kwargs.get("metabolite_outflow",{})
    solver = kwargs.get("solver",'gurobi')
    flobj = kwargs.get("flobj")
    endtime = kwargs.get("endtime",10**-2)
    media = kwargs.get("media",{})
    track_fluxes = kwargs.get("track_fluxes",True)
    save_internal_flux = kwargs.get("save_internal_flux",True) 
    resolution = kwargs.get("resolution",0.1)
    report_activity = kwargs.get("report_activity",True)
    report_activity_network= kwargs.get("report_activity_network",True)
    findwaves_report = kwargs.get("findwaves_report",False)
    debugging = kwargs.get("debugging",False)
    initial_abundance = kwargs.get("initial_abundance",None)
    refine_stoptime = kwargs.get("refine_intervals",False)



    start_time = time.time()


    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in community_members:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            if flnm.split(".")[-1] == "json":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.load_json_model(flnm)
            elif flnm.split(".")[-1] == "xml":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.read_sbml_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)


    
    try:
        flobj.write("[MetConSIN] Loaded " + str(len(cobra_models)) + " models successfully\n")
    except:
        print("[MetConSIN] Loaded " + str(len(cobra_models)) + " models successfully")


    if media == "minimal":
        for model in cobra_models.keys():
            mxg = cobra_models[model].slim_optimize()
            try:
                min_med = cb.medium.minimal_medium(cobra_models[model],mxg,minimize_components = True)
                cobra_models[model].medium = min_med
            except:
                pass
        kwargs["media"] = {}

    for mod in cobra_models.keys():
        cbgr = cobra_models[mod].slim_optimize()
        try:
            flobj.write("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))
        except:
            print("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))

    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,**kwargs)


    #for new network after perturbing metabolites, we only need to update mets0.
    #mets establishes an ordering of metabolites.
    #Next establish an ordering of microbes. Note, this won't necessarily be predictable, python
    #dict keys are unordered and calling dict.keys() will give whatever ordering it wants.

    y0 = np.array([y0dict[met] for met in metlist])

    model_list = [model for model in models.values()]
    if isinstance(initial_abundance,dict):
        x0 = np.array([initial_abundance.get(mod.Name,0) for mod in model_list])
    else:
        x0 = np.ones(len(model_list))


    inflow = np.zeros_like(y0)
    for ky,val in metabolite_inflow.items():
        try:
            i = list(metlist).index(ky)
        except ValueError:
            print("No {} in metlist, leaving inflow at 0. Did you forget a compartment tag?".format(ky))
            break
        inflow[i] = val

    outflow = np.zeros_like(y0)
    for ky,val in metabolite_outflow.items():
        try:
            i = list(metlist).index(ky)
        except ValueError:
            print("No {} in metlist, leaving outflow at 0. Did you forget a compartment tag?".format(ky))
            break
        outflow[i] = val


    dynamics = surf.surfin_fba(model_list,x0,y0,endtime,
                                inflow = inflow,
                                outflow = outflow,
                                solver = solver,
                                save_bases = True,
                                track_fluxes = track_fluxes,
                                save_internal_flux = save_internal_flux, 
                                resolution = resolution,
                                report_activity = report_activity, 
                                flobj = flobj,
                                fwreport = findwaves_report,
                                debugging=debugging,
                                refine_intervals = refine_stoptime)




    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Simulation & Bases computed in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Simulation & Bases computed in ",int(minuts)," minutes, ",sec," seconds.")

    # Put the dynamics into dataframes for easy saving/access
    x_sim = pd.DataFrame(dynamics["x"].round(7),columns = dynamics["t"],index = [model.Name for model in model_list])
    x_sim = x_sim.loc[:,~x_sim.columns.duplicated()]
    y_sim = pd.DataFrame(dynamics["y"].round(7),columns = dynamics["t"],index = metlist)
    y_sim = y_sim.loc[:,~y_sim.columns.duplicated()]
    basis_change_times = dynamics["bt"]

    all_sim = pd.concat([x_sim,y_sim])

    met_met_nets = {}
    mic_met_sum_nets = {}
    mic_met_nets = {}
    speciesHeuristic = {}
    interval_lens = {}
    total_interval = 0

    basis_change_info = pd.DataFrame(columns = np.unique(np.array(basis_change_times).round(6)),index = [model.Name for model in model_list]).fillna(0).astype(bool)
    ## To add: 
    ##         - details of network change - difference in edge strengths - dataframe for each model with columns=times index=metabolites
    ##                                                                    - seperate dataframe for incoming edges?
    ##                      Change "Combined" network to "average". Add "Combined" network with columns for each interval
    ##                                          Add "Difference" network with differnce in each edge at each basis change.

    basis_change_differences = dict([(mod.Name,{}) for mod in model_list])
    for model in model_list:
        dif = {}
        for i in range(1,len(dynamics["bases"][model.Name])):
            tp = dynamics["bases"][model.Name][i]
            ptp = dynamics["bases"][model.Name][i-1]
            tdi_act = {}
            tdi_act["Constraints"] = [get_constr_desc(j,model,metlist) for j in set(tp[1][0]).difference(set(ptp[1][0]))]
            tdi_act["Fluxes"] = [model.flux_order[j] for j in set(tp[1][1]).difference(set(ptp[1][1]))]
            tdi_dact = {}
            tdi_dact["Constraints"] = [get_constr_desc(j,model,metlist) for j in set(ptp[1][0]).difference(set(tp[1][0]))]
            tdi_dact["Fluxes"] = [model.flux_order[j] for j in set(ptp[1][1]).difference(set(tp[1][1]))]
            dif[tp[0]] = {"Activated":tdi_act,"Deactivated":tdi_dact}
        basis_change_differences[model.Name] = dif



    for i in range(len(basis_change_times)):
        #get the times - if we're on the last one there's no "end time"
        t0 = basis_change_times[i]
        try:
            t1 = basis_change_times[i+1]
            ky = "{:.4f}".format(t0)+"-"+"{:.4f}".format(t1)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))&(np.array(all_sim.columns.astype(float)).round(6)<=t1)]
            interval_lens[ky] = t1-t0
        except:
            ky = "{:.4f}-{:.4f}".format(t0,endtime)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))]
            total_interval = t0/(1-final_interval_weight)
            if t0 == 0:
                total_interval = 1
            interval_lens[ky] = total_interval-t0


        if dynamics_t.shape[1]:#skip if the interval is too small to contain any of the dynamics (this has to be due to roundoff error - there should be a point at each basis change time.)
            for model in model_list:
                #dynamics["basis"][model.Name] is a list of tuples of (basis change time, index of reduced basis - rows/columns)
                modbc = [bc[0] for bc in dynamics["bases"][model.Name]]#list of times this model changed basis
                if t0 in modbc:
                    basis_change_info.loc[model.Name,t0.round(6)] = True
                lastone = [indx for indx in range(len(modbc)) if modbc[indx] <= t0][-1]#this is the index in dynamics["basis"][model.Name] of the basis this model is using at this time interval

                #now we set the basis for the model to be the one it was using in this time interval
                if dynamics["bases"][model.Name][lastone] != None:
                    basinds = dynamics["bases"][model.Name][lastone][1]
                    Q,R = np.linalg.qr(model.standard_form_constraint_matrix[basinds[0]][:,basinds[1]])
                    model.current_basis = (Q,R,basinds)
                    bases_ok = True
                #if there isn't a basis for a model we quit.
                else:
                    bases_ok = False
                    break
            if bases_ok:
                metcons = y_sim.loc[:,t0].values
                # for model in model_list:
                #     print(model.current_basis[2])
                node_table,met_med_net,met_met_edges,met_met_nodes = mn.species_metabolite_network(metlist,metcons,model_list,report_activity=report_activity_network,flobj = flobj)
                #trim out species & metabolites that aren't present in this time interval.

                met_met_edges,met_met_nodes = mn.trim_network(met_met_edges,met_met_nodes,dynamics_t)
                met_med_net,node_table = mn.trim_network(met_med_net,node_table,dynamics_t)
                met_med_net_summary = mn.make_medmet_summ(met_med_net)
                ssnet,ssnodes,ssadj=mn.heuristic_ss(met_med_net_summary,node_table,report_activity=report_activity_network,flobj = flobj)

                met_met_nets[ky] = {"nodes":met_met_nodes,"edges":met_met_edges}
                mic_met_sum_nets[ky] = {"nodes":node_table,"edges":met_med_net_summary}
                mic_met_nets[ky] = {"nodes":node_table,"edges":met_med_net}
                speciesHeuristic[ky] = {"nodes":ssnodes,"edges":ssnet,"adjacency":ssadj}
    
    if total_interval == 0:
        print("Average network making - Total interval 0")
    else:
        for ky,val in interval_lens.items():
            interval_lens[ky] = val/total_interval
        
        avg_micmetnet_sum,comb_micmet_net_sum,avg_micmet_summ_nodes,rflag = mn.average_network(mic_met_sum_nets,interval_lens,"micmet")
        if rflag:
            mic_met_sum_nets["Average"] = {"nodes":avg_micmet_summ_nodes,"edges":avg_micmetnet_sum}
            diff_micmet_net_sum = make_diff_df(comb_micmet_net_sum)
            comb_micmet_net_sum["Source"] = [i.split("##")[0] for i in comb_micmet_net_sum.index]
            comb_micmet_net_sum["Target"] = [i.split("##")[1] for i in comb_micmet_net_sum.index]
            mic_met_sum_nets["Combined"] = {"nodes":avg_micmet_summ_nodes,"edges":comb_micmet_net_sum}
            mic_met_sum_nets["Difference"] = {"nodes":avg_micmet_summ_nodes,"edges":diff_micmet_net_sum}



        avg_micmetnet,comb_micmetnet,avg_micmet_nodes,rflag = mn.average_network(mic_met_nets,interval_lens,"micmet")
        if rflag:
            mic_met_nets["Average"] = {"nodes":avg_micmet_nodes,"edges":avg_micmetnet}
            diff_micmet_net = make_diff_df(comb_micmetnet)
            comb_micmetnet["Source"] = [i.split("##")[0] for i in comb_micmetnet.index]
            comb_micmetnet["Target"] = [i.split("##")[1] for i in comb_micmetnet.index]
            mic_met_nets["Combined"] = {"nodes":avg_micmet_nodes,"edges":comb_micmetnet}
            mic_met_nets["Difference"] = {"nodes":avg_micmet_nodes,"edges":diff_micmet_net}


        avg_metmetnet,comb_metmetnet,avg_metmet_nodes,rflag = mn.average_network(met_met_nets,interval_lens,"metmet")
        if rflag:
            met_met_nets["Average"] = {"nodes":avg_metmet_nodes,"edges":avg_metmetnet}
            diff_metmetnet = make_diff_df(comb_metmetnet)
            comb_metmetnet["Source"] = [i.split("##")[0] for i in comb_metmetnet.index]
            comb_metmetnet["Target"] = [i.split("##")[1] for i in comb_metmetnet.index]
            met_met_nets["Combined"] = {"nodes":avg_metmet_nodes,"edges":comb_metmetnet}
            met_met_nets["Difference"] = {"nodes":avg_metmet_nodes,"edges":diff_metmetnet}


        avg_spec,comb_spec,avg_spc_nodes,rflag = mn.average_network(speciesHeuristic,interval_lens,"spc")
        if rflag:
            speciesHeuristic["Average"] = {"nodes":avg_spc_nodes,"edges":avg_spec}
            diff_spec = make_diff_df(comb_spec)
            comb_spec["Source"] = [i.split("##")[0] for i in comb_spec.index]
            comb_spec["Target"] = [i.split("##")[1] for i in comb_spec.index]
            speciesHeuristic["Combined"] = {"nodes":avg_spc_nodes,"edges":comb_spec}
            speciesHeuristic["Difference"] = {"nodes":avg_spc_nodes,"edges":comb_spec}




    all_return = {"Microbes":x_sim,"Metabolites":y_sim,"SpeciesNetwork":speciesHeuristic,"MetMetNetworks":met_met_nets, "SpcMetNetworkSummaries":mic_met_sum_nets,"SpcMetNetworks":mic_met_nets, "BasisChanges":basis_change_info}
    bfcause = dynamics["bf"]
    all_return["InternalBasisDifferences"] = basis_change_differences

    basis_change_cause = {}
    for ky,val in bfcause.items():
        bc = {}
        for b in val:
            mky = b[0]
            fs = {}
            fs["Internal Flux Below 0"] = [models[mky].flux_order[j] for j in b[1]]
            fs["Constraint Violated"] = [get_constr_desc(j,models[mky],metlist) for j in b[2]]
            bc[mky] = fs
        basis_change_cause[ky] = bc


    all_return["BasisChangeCause"] = basis_change_cause
    if track_fluxes:
        exchg_fluxes = {}
        for model in model_list:
            try:
                eflux = pd.DataFrame(dynamics["Exchflux"][model.Name].round(7), columns = dynamics["t"],index = metlist)
                eflux = eflux.loc[:,~eflux.columns.duplicated()]
                exchg_fluxes[model.Name] = eflux
            except:
                pass
        all_return["ExchangeFluxes"] = exchg_fluxes

    if save_internal_flux:
        internal_flux = {}
        for model in model_list:
            try:
                total_interals = dynamics["Intflux"][model.Name].round(7)[:len(model.flux_order)] + dynamics["Intflux"][model.Name].round(7)[len(model.flux_order):2*len(model.flux_order)]
                iflux = pd.DataFrame(total_interals, columns = dynamics["t"],index = model.flux_order)
                iflux = iflux.loc[:,~iflux.columns.duplicated()]
                internal_flux[model.Name] = iflux
            except:
                pass
        all_return["InternalFluxes"] = internal_flux


    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Complete in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Complete in ",int(minuts)," minutes, ",sec," seconds.")

    return all_return

def save_metconsin(metconsin_return,flder):

    '''
    
    Saves the output of MetConSIN simulation in directory ``flder``. Creates subdirs for each network type with subdirs labeled by time interval containing a network (edges and nodes as seperate .tsv files)
    Also saves dynamics, with microbe and metabolite dynamics seperate, and plots them. Saves fluxes if those are included in the metconsin return

    :param metconsin_return: the output of metconsin_sim
    :type metconsin_return: dict 
    :param flder: desired path to output folder
    :type metconsin_return: dict 
    :return: None
    

    '''


    Path(flder).mkdir(parents=True, exist_ok=True)

    metconsin_return["Microbes"].to_csv(os.path.join(flder,"Microbes.tsv"),sep="\t")
    metconsin_return["Metabolites"].to_csv(os.path.join(flder,"Metabolites.tsv"),sep="\t")

    with open(os.path.join(flder,"BasisChangeCauses.json"),'w') as fl:
        json.dump(metconsin_return["BasisChangeCause"],fl, default=to_serializable)
    with open(os.path.join(flder,"InternalBasisDifferences.json"),'w') as fl:
        json.dump(metconsin_return["InternalBasisDifferences"],fl, default=to_serializable)

    pd.DataFrame(metconsin_return["BasisChanges"]).to_csv(os.path.join(flder,"BasisTimes.tsv"),sep = "\t")

    metmic_folder = os.path.join(flder,"SpcMetNetworks")
    metmet_folder = os.path.join(flder,"MetaboliteNetworks")
    micmic_folder = os.path.join(flder,"SpeciesNetworks")

    Path(metmic_folder).mkdir(parents=True, exist_ok=True)
    Path(metmet_folder).mkdir(parents=True, exist_ok=True)
    Path(micmic_folder).mkdir(parents=True, exist_ok=True)



    for ky in metconsin_return["MetMetNetworks"].keys():
        if ky not in ["Combined","Average","Difference"]:
            Path(os.path.join(metmet_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["MetMetNetworks"][ky]["edges"].to_csv(os.path.join(metmet_folder,ky,"MetMetEdges"+ky+".tsv"),sep="\t")
            metconsin_return["MetMetNetworks"][ky]["nodes"].to_csv(os.path.join(metmet_folder,ky,"MetMetNodes"+ky+".tsv"),sep="\t")
            Path(os.path.join(metmic_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["SpcMetNetworkSummaries"][ky]["edges"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksEdgesSummary"+ky+".tsv"),sep="\t")
            metconsin_return["SpcMetNetworks"][ky]["edges"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksEdges"+ky+".tsv"),sep="\t")
            metconsin_return["SpcMetNetworks"][ky]["nodes"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksNodes"+ky+".tsv"),sep="\t")
            Path(os.path.join(micmic_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["SpeciesNetwork"][ky]["edges"].to_csv(os.path.join(micmic_folder,ky,"SpeciesNetworkEdges"+ky+".tsv"),sep="\t")
            metconsin_return["SpeciesNetwork"][ky]["nodes"].to_csv(os.path.join(micmic_folder,ky,"SpeciesNetworkNodes"+ky+".tsv"),sep="\t")
        else:
            # {"nodes":avg_micmet_summ_nodes,"edges":avg_micmetnet_sum}
            Path(os.path.join(metmet_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["MetMetNetworks"][ky]["edges"].to_csv(os.path.join(metmet_folder,ky,"MetMetEdgesAvg.tsv"),sep="\t")
            metconsin_return["MetMetNetworks"][ky]["nodes"].to_csv(os.path.join(metmet_folder,ky,"MetMetNodes.tsv"),sep="\t")
            Path(os.path.join(metmic_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["SpcMetNetworkSummaries"][ky]["edges"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksEdgesSummaryAvg.tsv"),sep="\t")
            metconsin_return["SpcMetNetworks"][ky]["edges"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksEdgesAvg.tsv"),sep="\t")
            metconsin_return["SpcMetNetworks"][ky]["nodes"].to_csv(os.path.join(metmic_folder,ky,"SpcMetNetworksNodesAvg.tsv"),sep="\t")
            Path(os.path.join(micmic_folder,ky)).mkdir(parents=True, exist_ok=True)
            metconsin_return["SpeciesNetwork"][ky]["edges"].to_csv(os.path.join(micmic_folder,ky,"SpeciesNetworkEdgesAvg.tsv"),sep="\t",index = False)
            metconsin_return["SpeciesNetwork"][ky]["nodes"].to_csv(os.path.join(micmic_folder,ky,"SpeciesNetworkNodesAvg.tsv"),sep="\t")

    if "ExchangeFluxes" in metconsin_return.keys():


        exchange_flux_flder = os.path.join(flder,"ExchangeFluxes")

        Path(exchange_flux_flder).mkdir(parents=True, exist_ok=True)

        for model in metconsin_return["ExchangeFluxes"].keys():
            metconsin_return["ExchangeFluxes"][model].to_csv(os.path.join(exchange_flux_flder,model.replace(".","")+"exchange_flux.tsv"),sep="\t")

    if "InternalFluxes" in metconsin_return.keys():

        internal_flux_flder = os.path.join(flder,"InternalFluxes")

        Path(internal_flux_flder).mkdir(parents=True, exist_ok=True)

        for model in metconsin_return["InternalFluxes"].keys():
            metconsin_return["InternalFluxes"][model].to_csv(os.path.join(internal_flux_flder,model.replace(".","")+"internal_flux.tsv"),sep="\t")




    x_pl = metconsin_return["Microbes"].copy()
    x_pl.columns = np.array(metconsin_return["Microbes"].columns).astype(float).round(4)
    ax = x_pl.T.plot(figsize = (20,10))
    for bt in metconsin_return["BasisChanges"].columns:
        ax.axvline(x = bt,linestyle = ":")
    ax.legend(prop={'size': 30},loc = 2)
    plt.savefig(os.path.join(flder,"Microbes.png"))
    plt.close()

    nonzero_mets = np.array([max(metconsin_return["Metabolites"].loc[nd]) >10**-6 for nd in metconsin_return["Metabolites"].index])

    y_pl = metconsin_return["Metabolites"].copy()
    y_pl.columns = np.array(metconsin_return["Metabolites"].columns).astype(float).round(4)
    if sum(nonzero_mets):
        ax = y_pl.loc[nonzero_mets].T.plot(figsize = (20,10))
    else:
        ax = y_pl.T.plot(figsize = (20,10))
    
    for bt in metconsin_return["BasisChanges"].columns:
        ax.axvline(x = bt,linestyle = ":")
    ax.legend(prop={'size': 15})
    plt.savefig(os.path.join(flder,"Metabolites.png"))
    plt.close()

    for model in metconsin_return["ExchangeFluxes"].keys():
        exchpl = metconsin_return["ExchangeFluxes"][model].copy()
        exchpl.columns = np.array(metconsin_return["ExchangeFluxes"][model].columns).astype(float).round(4)
        if sum(nonzero_mets):
            ax = exchpl.loc[nonzero_mets].T.plot(figsize = (20,10))
        else:
            ax = exchpl.T.plot(figsize = (20,10))
        for bt in metconsin_return["BasisChanges"].columns:
            ax.axvline(x = bt,linestyle = ":")
        ax.legend(prop={'size': 15})
        plt.savefig(os.path.join(exchange_flux_flder,model + "Exchange.png"))
        plt.close()

def dynamic_fba(community_members,model_info_file,initial_abundance,**kwargs):

    '''

    Does Full Simulation only, does not make networks. Requires initial community composition (treats as absolute). Initial metabolites can be modified with the ``media`` parameter.

    :param community_members: list of models of taxa in the community of interest
    :type community_members: list[str]
    :param model_info_file: path to csv with GEM info. csv file should contain a column called "Species" with names that include the names in ``desired_models``, and a column called "File" that contains the path to a GSM in xml or json form (note that .mat is not currently supported).
    :type model_info_file: str

    :param initial_abundance: intial abundances (treated as absolute) of the community to be simulated. Should be a dictionary keyed by community members.
    :type initial_abundance: dict[str,float]


    :param save_tsvs: Whether or not to save the dynamics as .tsv files. Default False
    :type save_tsvs: bool
    :param save_folder: desired path to folder with saved .tsv files. Default DFBA
    :type save_folder: str
    :param endtime: Simulation length. Default 10**-2
    :type endtime: float
    :param resolution: Time-resolution of the dynamics output. Default 0.1
    :type resolution: float
    :param media: Growth media used for the simulation. If None, averages the media dicts packaged with the GSMs used. Can be keyed by metabolite name, ID, or exchange reaction ID. Alternatively, passing the string "minimal" attempts to define a minimal growth media for each microbe and averages that. Default None.
    :type media: dict or str
    :param metabolite_inflow: Inflow rate for each metabolite. Default all 0
    :type metabolite_inflow: array[float]
    :param metabolite_outflow: Outflow rate for each metabolite. Default all 0
    :type metabolite_outflow: array[float]
    :param model_deathrates: Decay rate for each community member, given as dictionary keyed by community member names. Defaul all 0
    :type model_deathrates: dict[str,float]


    :param ub_funs: General function to use for upper bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``.  Default ``linear``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type ub_funs: str
    :param lb_funs: General function to use for lower bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``. Default ``constant``. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`.
    :type lb_funs: str
    :param ub_params: Parameters to use for upper bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type ub_params: dict
    :param lb_params: Parameters to use for lower bound functions. See :py:func:`prep_cobrapy_models <prep_models.prep_cobrapy_models>`..
    :type lb_params: dict

    :param upper_bound_user_functions: User-defined upper bound functions for exchange reactions. **Ignored** unless ``ub_funs`` == ``user``. Any model not included (which can be all models), will default to functions defined by the ``ub_funs`` parameter. Default None.
    :type upper_bound_user_functions: dict[str,array of lambdas]
    :param lower_bound_user_functions: User-defined lower bound functions for exchange reactions. **Ignored** unless ``lb_funs`` == ``user``.Any model not included (which can be all models), will default to functions defined by the ``lb_funs`` parameter. Default None.
    :type lower_bound_user_functions: dict[str,array of lambdas]    
    :param upper_bound_user_functions_dt: Derivatives of user defined upper_bound_functions_dt. Default None
    :type upper_bound_user_functions_dt:  dict[str,array of lambdas]
    :param lower_bound_user_functions_dt: kwargs.get(lower_bound_functions_dt)
    :type lower_bound_user_functions_dt:  dict[str,array of lambdas]



    :param solver: LP solver to use (currently supports ``gurobi`` and ``clp``). Default ``gurobi``
    :type solver: str
    :param met_filter: If ``met_filter_sense`` == "exclude", list of metabolites to treat as infinitely supplied (i.e. ignored in the dynamics). If ``met_filter_sense`` == "include", all other metabolites will be treated as infinitely supplied (i.e. ignored in the dynamics). Default None
    :type met_filter: list
    :param met_filter_sense: Choice of "include" or "exclude" for metabolite filter. If left as None, no filter will be done. Default None
    :type met_filter: str
    :param track_fluxes: Whether or not to save the exchange fluxes computed during the simulation. Default True
    :type track_fluxes: bool
    :param save_internal_flux: Whether or not to save the internal fluxes computed during the simulation. Default True
    :type save_internal_flux: bool
    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File
    :param report_activity: Whether or not to log simulation progress. Default True
    :type report_activity: bool
    :param findwaves_report: Whether or not to log details of basis finding. Default False
    :type findwaves_report: bool
    :param debugging: Turn on some debugging prints. Default False
    :type debugging: bool
    :param forceOns: Whether or not to allow internal reactions to be forced on (with a positive lower bound). Many GSM include such bounds. Default True
    :type forceOns: bool

    .. note::

        When setting bound functions and parameters, any models left out of a parameter dictionary will use default parameters for the choice of function type. Currently, the only way to use different function **types** for each model is to use user defined functions. 
        
    .. warning::

        When creating user defined functions, pay attention to issues with variable scope! See `python docs <https://docs.python.org/3.4/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result>`_ for more information.


    :return: Dictionary containing the simulation and networks. Keys are :

    - *Microbes*\ : dynamics of the microbial taxa, as a pandas dataframe
    - *Metabolites*\ : dynamics of the metabolites, as a pandas dataframe
    - *BasisChanges*\ : Times that the system updated a basis
    - *ExchangeFluxes*\ (if ``track_fluxes`` == True): Exchange fluxes at each time-point for each taxa.
    - *InternalFluxes*\ (in ``save_internal_flux`` == True): Internal fluxes at each time-point for each taxa.

    :rtype: dict 

    '''

    

    metabolite_inflow = kwargs.get("metabolite_inflow")
    metabolite_outflow = kwargs.get("metabolite_outflow")
    solver = kwargs.get("solver",'gurobi')
    flobj = kwargs.get("flobj")
    endtime = kwargs.get("endtime",10**-2)
    media = kwargs.get("media",{})
    track_fluxes = kwargs.get("track_fluxes",True)
    save_internal_flux = kwargs.get("save_internal_flux",True) 
    resolution = kwargs.get("resolution",0.1)
    report_activity = kwargs.get("report_activity",True)
    forceOns = kwargs.get("forceOns",True)
    findwaves_report = kwargs.get("findwaves_report",False)
    debugging = kwargs.get("debugging",False)
    save_tsvs = kwargs.get("save_tsvs",False)
    save_folder = kwargs.get("save_folder","DFBA")



    start_time = time.time()





    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in community_members:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            if flnm.split(".")[-1] == "json":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.load_json_model(flnm)
            elif flnm.split(".")[-1] == "xml":
                with contextlib.redirect_stderr(None):
                    cobra_models[mod] = cb.io.read_sbml_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)


    
    try:
        flobj.write("[MetConSIN] Loaded " + str(len(cobra_models)) + " models successfully\n")
    except:
        print("[MetConSIN] Loaded " + str(len(cobra_models)) + " models successfully")


    if media == "minimal":
        for model in cobra_models.keys():
            mxg = cobra_models[model].slim_optimize()
            try:
                min_med = cb.medium.minimal_medium(cobra_models[model],mxg,minimize_components = True)
                cobra_models[model].medium = min_med
            except:
                pass
        kwargs["media"] = {}


    for mod in cobra_models.keys():
        cbgr = cobra_models[mod].slim_optimize()
        try:
            flobj.write("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))
        except:
            print("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))

    #returns dict of surfmods, list of metabolites, and concentration of metabolites.


    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,**kwargs)

    #for new network after perturbing metabolites, we only need to update mets0.
    #mets establishes an ordering of metabolites.
    #Next establish an ordering of microbes. Note, this won't necessarily be predictable, python
    #dict keys are unordered and calling dict.keys() will give whatever ordering it wants.

    y0 = np.array([y0dict[met] for met in metlist])

    model_list = [model for model in models.values()]


    x0 = np.array([initial_abundance.get(mod,0) for mod in model_list])


    dynamics = surf.surfin_fba(model_list,x0,y0,endtime,
                                inflow = metabolite_inflow,
                                outflow = metabolite_outflow,
                                solver = solver,
                                save_bases = True,
                                track_fluxes = track_fluxes,
                                save_internal_flux = save_internal_flux, 
                                resolution = resolution,
                                report_activity = report_activity, 
                                flobj = flobj,
                                fwreport = findwaves_report,
                                debugging=debugging)




    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Simulation & Bases computed in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Simulation & Bases computed in ",int(minuts)," minutes, ",sec," seconds.")

    # Put the dynamics into dataframes for easy saving/access
    x_sim = pd.DataFrame(dynamics["x"].round(7),columns = dynamics["t"],index = [model.Name for model in model_list])
    x_sim = x_sim.loc[:,~x_sim.columns.duplicated()]
    y_sim = pd.DataFrame(dynamics["y"].round(7),columns = dynamics["t"],index = metlist)
    y_sim = y_sim.loc[:,~y_sim.columns.duplicated()]
    basis_change_times = dynamics["bt"]

    all_return = {"Microbes":x_sim,"Metabolites":y_sim,"BasisChanges":basis_change_times}

    if save_tsvs:
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        x_sim.to_csv(os.path.join(save_folder,"Microbes.tsv"),sep="\t")
        y_sim.to_csv(os.path.join(save_folder,"Metabolites.tsv"),sep="\t")
        pd.DataFrame(basis_change_times).to_csv(os.path.join(save_folder,"BasisTimes.tsv"),sep = "\t")

    if track_fluxes:
        exchg_fluxes = {}
        for model in model_list:
            try:
                eflux = pd.DataFrame(dynamics["Exchflux"][model.Name].round(7), columns = dynamics["t"],index = metlist)
                eflux = eflux.loc[:,~eflux.columns.duplicated()]
                exchg_fluxes[model.Name] = eflux
            except:
                pass
        if save_tsvs:
            exchange_flux_flder = os.path.join(save_folder,"ExchangeFluxes")

            Path(exchange_flux_flder).mkdir(parents=True, exist_ok=True)

            for model in exchg_fluxes.keys():
                exchg_fluxes[model].to_csv(os.path.join(exchange_flux_flder,model.replace(".","")+"exchange_flux.tsv"),sep="\t")

        all_return["ExchangeFluxes"] = exchg_fluxes

    if save_internal_flux:
        internal_flux = {}
        for model in model_list:
            try:
                total_interals = dynamics["Intflux"][model.Name].round(7)[:len(model.flux_order)] + dynamics["Intflux"][model.Name].round(7)[len(model.flux_order):2*len(model.flux_order)]
                iflux = pd.DataFrame(total_interals, columns = dynamics["t"],index = model.flux_order)
                iflux = iflux.loc[:,~iflux.columns.duplicated()]
                internal_flux[model.Name] = iflux
            except:
                pass

        if save_tsvs:
            internal_flux_flder = os.path.join(save_folder,"InternalFluxes")

            Path(internal_flux_flder).mkdir(parents=True, exist_ok=True)

            for model in exchg_fluxes.keys():
                exchg_fluxes[model].to_csv(os.path.join(internal_flux_flder,model.replace(".","")+"internal_flux.tsv"),sep="\t")


        all_return["InternalFluxes"] = internal_flux


    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Complete in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Complete in ",int(minuts)," minutes, ",sec," seconds.")

    return all_return

def get_constr_desc(j,mod,metablist):

    '''Creates a description of constraint j in model
    
    :param j: constraint number in ordering of model.prob_mat
    :type j: int
    :param mod: model of interest
    :type mod: SurfMod
    :param metablist: list of exchanged metablites
    :type metablist: listlike

    :return: description of constraint
    :rtype: str
    
    '''

    if j < mod.num_exch_rxns:
        mj = mod.ExchangeOrder[j]
        met = metablist[mj]
        return "{} exchange upper bound".format(met)
    elif mod.num_exch_rxns <= j < 2*mod.num_exch_rxns:
        mj = mod.ExchangeOrder[j-mod.num_exch_rxns]
        met = metablist[mj]
        return "{} exchange lower bound".format(met)
    elif 2*mod.num_exch_rxns <= j < 2*mod.num_exch_rxns + mod.num_fluxes:
        return "{} upper bound".format(mod.flux_order[j-2*mod.num_exch_rxns])
    else:
        tj = j - 2*mod.num_exch_rxns + mod.num_fluxes
        if tj < mod.num_internal_metabolites:
            return "{} equilibrium".format(mod.internal_metabolites[tj])
        else:
            return "{} equilibrium".format(mod.internal_metabolites[tj - mod.num_internal_metabolites])
        
def make_diff_df(df):

    '''
    Makes the ``Difference`` networks table.

    :param df: DataFrame indexed by edges in the network set, with a column for each time interval
    :type df: pd.DataFrame

    :return: DataFrame indexed by edges in the network set, with a column for each transition. Values are old network edge weight minus new.
    :rtype: pd.DataFrame
    
    '''

    ddf = pd.DataFrame(index = df.index,columns = [col.split("-")[1] for col in df.columns])
    for col in ddf.columns:
        starts = [c for c in df.columns if c.split("-")[0] == col]
        if len(starts):
            c1 = starts[0]
            c2 = [c for c in df.columns if c.split("-")[1] == col][0]
            ddf[col] = df[c1]-df[c2]
        else:
            ddf.drop(col,axis =1,inplace = True)
    ddf["Source"] = [i.split("##")[0] for i in ddf.index]
    ddf["Target"] = [i.split("##")[1] for i in ddf.index]
    return ddf