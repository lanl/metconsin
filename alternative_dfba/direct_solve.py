import numpy as np
import cobra as cb
import pandas as pd
import time
import contextlib
import sys
import os
from scipy.integrate import solve_ivp

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from metconsin import prep_models as pr
from metconsin import make_network as mn



### Use euler method for simulation. Can also make networks in a very simple way by taking the sign of the products of exchange reactions. Average over time?
## Want it to be the same input as metconsin. Output will be different (doesn't have good detailed networks).

def direct_solve(community_members,model_info_file,**kwargs):

    '''

    Does Full Simulation, makes network based on average sign with shared metabolites.

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

    .. note::

        When setting bound functions and parameters, any models left out of a parameter dictionary will use default parameters for the choice of function type. Currently, the only way to use different function **types** for each model is to use user defined functions. 
        
    .. warning::

        When creating user defined functions, pay attention to issues with variable scope! See `python docs <https://docs.python.org/3.4/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result>`_ for more information.


    :return: Dictionary containing the simulation and networks. Keys are :

    - *Microbes*\ : dynamics of the microbial taxa, as a pandas dataframe
    - *Metabolites*\ : dynamics of the metabolites, as a pandas dataframe
    - *SpeciesNetwork*\ : Species-Species networks defined by simple hueristic (keyed by time point)
    - *SpcMetNetworks*\ : Microbe-Metabolite networks defined by simple hueristic (keyed by time point)


    :rtype: dict 

    '''


    metabolite_inflow = kwargs.get("metabolite_inflow",{})
    metabolite_outflow = kwargs.get("metabolite_outflow",{})
    flobj = kwargs.get("flobj")
    endtime = kwargs.get("endtime",10**-2)
    media = kwargs.get("media",{})
    resolution = kwargs.get("resolution",0.1)
    initial_abundance = kwargs.get("initial_abundance",None)
    deathrates = kwargs.get("model_deathrates")
    networktimes = kwargs.get("network_times",[])

    if not len(networktimes):
        networktimes = [0,endtime/2,endtime]



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




    ### NEED TO MAKE Y0 FROM MEDIA, RECONCILE WITH REACTION IDS IN COBRA MODELS
    surfmodels,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,**kwargs) #does more than I need...

    if deathrates == None:
        deathrates = dict([(modky,0) for modky in surfmodels.keys()])

    model_list = [model for model in surfmodels.values()]
    if isinstance(initial_abundance,dict):
        x0 = np.array([initial_abundance.get(mod.Name,0) for mod in model_list])
    else:
        x0 = np.ones(len(model_list))

    # gives me a dict of models in surfmod form, list of metabolite names in order, and array of initial metabolite concentration from the media
    met_filter = kwargs.get("met_filter",[])
    met_filter_sense = kwargs.get("met_filter_sense","exclude")
    ex_rxns = {}

    for nm,model in cobra_models.items():
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#


        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exchng_metabolite_ids = [t[1] for t in exchng_metabolite_ids_wrx]
        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name for metab in exchng_metabolite_ids]

        #filter out metabolites we want to ignore.
        if len(met_filter):
            if met_filter_sense == "exclude":
                exchng_metabolite_names_flt = [met for met in exchng_metabolite_names if met not in met_filter]
            elif met_filter_sense == "include":
                exchng_metabolite_names_flt = [met for met in exchng_metabolite_names if met in met_filter]
            else:
                exchng_metabolite_names_flt = exchng_metabolite_names
                print("[prep_cobrapy_models] Must specify sense of metabolite filter - exclude or include.\n No filtering done.")
            exchng_metabolite_ids_flt = [metid for metid in exchng_metabolite_ids if model.metabolites.get_by_id(metid).name in exchng_metabolite_names_flt]
            exchng_reactions_flt = [rxid for rxid in exchng_reactions if all([metid in exchng_metabolite_ids_flt for metid in model.reactions.get_by_id(rxid).reactants])]

            exchng_metabolite_names = exchng_metabolite_names_flt
            exchng_reactions = exchng_reactions_flt

        #to insure proper ordering, need EX[i] to exchange metabolite j where surfmod[nm].ExchangeOrder[i] = j
        metaborder = [metlist[j] for j in surfmodels[nm].ExchangeOrder]
        ex_rxns[nm] = [exchng_reactions[exchng_metabolite_names.index(m)] for m in metaborder]

    y0 = np.array([y0dict[met] for met in metlist])

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

    feasibility_check.terminal = True

    cobramods = [cobra_models[sp] for sp in community_members]
    surfmods = [surfmodels[sp] for sp in community_members]
    exchng_ids = [ex_rxns[sp] for sp in community_members]
    deathrates = [deathrates[sp] if sp in deathrates.keys() else 0 for sp in community_members]

    sol = solve_ivp(
        fun=dfba_system,
        args=(cobramods,surfmods,exchng_ids,inflow,outflow,deathrates),
        events=[feasibility_check],
        t_span=(0,endtime),
        t_eval= np.arange(0,endtime,resolution),
        y0=np.concatenate([x0,y0]),
        rtol=1e-6,
        atol=1e-8,
        method='BDF')
    

    microbes = pd.DataFrame(sol.y[:len(community_members)].round(7),index = community_members,columns = sol.t)
    metabolites = pd.DataFrame(sol.y[len(community_members):].round(7),index = metlist,columns = sol.t)

    spc_met_nets = {}
    spc_spc_nets = {}
    for tm in networktimes:
        i = np.where(sol.t <= tm)[0][-1]
        spmet,nodes = infer_network(sol.y[:,i],cobramods,surfmods,exchng_ids,community_members,metlist)
        spsp,_,_= mn.heuristic_ss(spmet,nodes,report_activity = False,flobj = None)
        spc_met_nets[tm] = spmet
        spc_spc_nets[tm] = spsp

    return {"Microbes":microbes,"Metabolites":metabolites,"SpcMetNetworks":spc_met_nets,"SpeciesNetwork":spc_spc_nets,"SolverStatus":sol.status}



def make_new_media(y,surfmod,exrs):
    mod_y = y[surfmod.ExchangeOrder]
    mod_exbds = [bd(mod_y) for bd in surfmod.exchange_bounds]
    new_media = {}
    for i,rid in enumerate(exrs):
        new_media[rid] = mod_exbds[i]
    return new_media

def get_flux(n,sfmod,exr,optfl):
    loc = np.array(exr)[np.where(sfmod.ExchangeOrder == n)]
    if len(loc):
        try:
            return optfl[loc].mean()
        except:
            return 0
    else:
        return 0

def dfba_system(t,z,cobramods,surfmods,exchng_ids,inflow,outflow,deathrates):
    #cobramods and surfmods and exchng_ids as lists instead of dicts
    x = z[:len(surfmods)]
    y = z[len(surfmods):]
    dxdt = -x*np.array(deathrates)
    dydt = np.array(inflow) - y*np.array(outflow)
    for i,cmod in enumerate(cobramods):
        cmod.medium = make_new_media(y,surfmods[i],exchng_ids[i])
        opt = cmod.optimize()
        if opt.status != 'infeasible':
            dxdt[i] += opt.objective_value*x[i]
            dydt += x[i]*np.array([get_flux(j,surfmods[i],exchng_ids[i],opt) for j in range(len(y))])
    return np.concatenate([dxdt,dydt])

def feasibility_check(t,z,cobramods,surfmods,exchng_ids,inflow,outflow,deathrates):
    #cobramods and surfmods and exchng_ids as lists instead of dicts
    x = z[:len(surfmods)]
    y = z[len(surfmods):]
    f = 0
    for i,cmod in enumerate(cobramods):
        with cmod:
            cmod.medium = make_new_media(y,surfmods[i],exchng_ids[i])
            cb.util.add_lp_feasibility(cmod)
            feasibility = cb.util.fix_objective_as_constraint(cmod)
            f += abs(feasibility)
    return feasibility - 10**-6

def infer_network(z,cobramods,surfmods,exchng_ids,modnames,metlist):
    #cobramods and surfmods and exchng_ids as lists instead of dicts
    y = z[len(surfmods):]
    net = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    nodes = pd.DataFrame(columns = ["Type","Name"])
    for m in metlist:
        nodes.loc[m] = ["Metabolite",m]
    for i,cmod in enumerate(cobramods):
        nodes.loc[modnames[i]] = ["Microbe",modnames[i]]
        cmod.medium = make_new_media(y,surfmods[i],exchng_ids[i])
        opt = cmod.optimize()
        if opt.status != 'infeasible':
            y_dt = np.array([np.round(get_flux(j,surfmods[i],exchng_ids[i],opt),8) for j in range(len(y))])
            for j,m in enumerate(metlist):
                if y_dt[j]:
                    net.loc["{}##{}".format(modnames[i],m)] = [modnames[i],m,"Microbe",y_dt[j],np.abs(y_dt[j]),np.sign(y_dt[j]),np.sqrt(np.abs(y_dt[j])),np.sign(y_dt[j])*np.sqrt(np.abs(y_dt[j]))]
                if y_dt[j] < 0:
                    net.loc["{}##{}".format(m,modnames[i])] = [m,modnames[i],"Metabolite",-y_dt[j],np.abs(y_dt[j]),-np.sign(y_dt[j]),np.sqrt(np.abs(y_dt[j])),-np.sign(y_dt[j])*np.sqrt(np.abs(y_dt[j]))]
    return net,nodes