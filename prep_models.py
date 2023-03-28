import numpy as np
import scipy as sp
from surfmod import *
import pandas as pd

def prep_cobrapy_models(models,
                        forceOns = True,
                        upper_bound_functions = None,
                        lower_bound_functions = None,
                        upper_bound_functions_dt = None,
                        lower_bound_functions_dt = None, 
                        media = None, 
                        met_filter = None,
                        met_filter_sense = "exclude", 
                        lb_funs = "constant", 
                        ub_funs = "linearRand",
                        linearScale = 1.0,
                        flobj = None,
                        deathrates = None):


    """
    
    Creates a set of SurfMod objects from a set of cobrapy GSMs, and interprets the media (either given by the user or inferred from the cobrapy models) to create an initial metbaolite concentration. 

    :param media: Growth media used for the simulation. If None, averages the media dicts packaged with the GSMs used. Can be keyed by metabolite name, ID, or exchange reaction ID. Alternatively, passing the string "minimal" attempts to define a minimal growth media for each microbe and averages that. Default None.
    :type media: dict or str

    :param upper_bound_functions: Upper bound functions for exchange reactions. Can be strings with same options as ``ub_funs`` or dicts of user defined functions. Any model not included (which can be all models), will default to functions defined by the ``ub_funs`` parameter. Default None.
    :type upper_bound_functions: dict[str] or dict[dict[function]]
    :param ub_funs: General function to use for upper bounds on exchange reactions. Options are ``constant``, ``linearRand``, ``linearScale``, ``hill1Rand``, ``hill11``.  Default ``linearRand``
    :type ub_funs: str
    :param lower_bound_functions: Lower bound functions for exchange reactions. Any model not included (which can be all models), will default to functions defined by the ``lb_funs`` parameter. Default None.
    :type lower_bound_functions: dict[str] or dict[dict[function]]    
    :param lb_funs: General function to use for lower bounds on exchange reactions. Options are ``constant``, ``linearRand``, ``linearScale``, ``hill1Rand``, ``hill11``. Default ``constant``
    :type lb_funs: str
    :param upper_bound_functions_dt: Derivatives of user defined upper_bound_functions_dt. Default None
    :type upper_bound_functions_dt: dict[dict[function]]
    :param lower_bound_functions_dt: kwargs.get(lower_bound_functions_dt)
    :type lower_bound_functions_dt: dict[dict[function]]
    :param linearScale: Uniform coefficient for exchange reaction bounds if ``lb_funs`` or ``ub_funs`` (or entries in ``upper_bound_functions`` or ``lower_bound_functions``) are "linearScale". Default 1.0
    :type linearScale: float

    :param deathrates: Decay rate for each community member, given as dictionary keyed by community member names. Defaul all 0
    :type deathrates: dict[str,float]

    :param met_filter: If ``met_filter_sense`` == "exclude", list of metabolites to treat as infinitely supplied (i.e. ignored in the dynamics). If ``met_filter_sense`` == "include", all other metabolites will be treated as infinitely supplied (i.e. ignored in the dynamics). Default None
    :type met_filter: list
    :param met_filter_sense: Choice of "include" or "exclude" for metabolite filter. If left as None, no filter will be done. Default None
    :type met_filter: str

    :param forceOns: Whether or not to allow internal reactions to be forced on (with a positive lower bound). Many GSM include such bounds. Default True
    :type forceOns: bool

    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File

    :return: Tuple of SurfMods, metabolite names, and intitial metabolite concentrations
    :rtype: tuple[list[SurfMods],list[str],array[float]]

    Upper and lower bound functions can be user defined seperately for each bound and each model with a dict (keyed by model name) of dicts (keyed by metabolite name). Values in these dicts can be functions (in which case the 
    user must provide the derivative of the function with respect to time in ``upper_bound_functions_dt`` or ``lower_bound_functions_dt``) or a choice of predifined functions (see below). Alternatively, functions can be chosen
    for each model using the predefined function with a dict (keyed by model name) of options from the predefined list. Finally, functions can be chosen to be the same for each model by passing an option from the predefined functions
    to ``ub_funs`` and/or ``lb_funs`` and passing None to ``upper_bound_functions`` and/or ``lower_bound_functions``.

    The predifined options are:

    - *constant*\ : Bounds do not depend on the metabolites. 
    - *linearRand*\ : Bounds are linear in the corresponding metabolite with a random constant of proportionality
    - *linearScale*\ : Bounds are linear in teh corresponding metabolite with a uniform constant given by ``linearScale``
    - *hill1Rand*\ : Bounds follow a Hill function of the corresponding metabolite with exponent 1 and random Kd
    - *hill11*\ : Bounds follow a Hill function of the corresponding metabolite with exponent 1 and Kd 1

    """


    #can provide metabolite uptake dictionary as dict of dicts {model_key1:{metabolite1:val,metabolite2:val}}

                        # upper_bound_functions = {},
                        # lower_bound_functions = {},
                        # upper_bound_functions_dt = {},
                        # lower_bound_functions_dt = {}, 
                        # media = {}, 
                        # met_filter = [],

    if media == None:
        media = {}
    if upper_bound_functions == None:
        upper_bound_functions = {}
    if lower_bound_functions == None:
        lower_bound_functions = {}
    if upper_bound_functions_dt == None:
        upper_bound_functions_dt = {}
    if lower_bound_functions_dt == None:
        lower_bound_functions_dt = {}
    if met_filter == None:
        met_filter = []
    if deathrates == None:
        deathrates = dict([(modky,0) for modky in models.keys()])


    from cobra import util

    if not isinstance(models,dict):
        modeldict = {}
        for mod in models:
            modeldict[mod.name] = mod
        models = modeldict

    metaabs = {}
    y0s = {}
    exrn = {}
    metabids = {}
    nametoid = {}
    exrn_to_exmet = {}
    exmet_to_exrn = {}

    for modelkey in models.keys():
        model = models[modelkey]

        #list all reactions the model claims are exchange.
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#


        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exrxn_to_met = dict(exchng_metabolite_ids_wrx)
        met_to_exrxn = dict([(t[1],t[0]) for t in exchng_metabolite_ids_wrx])
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
            exchng_metabolite_ids = exchng_metabolite_ids_flt
            exchng_reactions = exchng_reactions_flt



        idtonm = dict(zip(exchng_metabolite_ids,exchng_metabolite_names))
        nmtoid = dict(zip(exchng_metabolite_names,exchng_metabolite_ids))

        if len(media):#if we're given a media, I'm assuming it's keyed by metabolite name, ID, or exchange reaction ID. IF THIS IS THE CASE, THE KEYS MUST MATCH METABOLITE NAMES or METABOLITE IDS or REACTION IDs THAT ARE IN THE MODEL.
            environment = media
        else:#otherwise the media file associated with the model is keyed by reaction ID
            environment = {}
            for rxn in model.medium:
                for met in model.reactions.get_by_id(rxn).reactants:
                    environment[met.id] = model.medium[rxn]

        y_init = {}
        for metabo in exchng_metabolite_names:
            if metabo in environment.keys():#environment is keyed by metabolite name.
                y_init[metabo] = environment[metabo]
            elif nmtoid[metabo] in environment.keys():#environment is keyed by metabolite ID
                y_init[metabo] = environment[nmtoid[metabo]]
            elif any([nmtoid[metabo] in rx.reactants for rx in model.reactions]):#environment is keyed by exchange reaction ID
                y_init[metabo] = np.mean([environment[rx.id] for rx in model.reactions if metabo in rx.reactants])




        metaabs[model.name] = exchng_metabolite_names
        metabids[model.name] = exchng_metabolite_ids
        y0s[model.name] = y_init
        exrn[model.name] = exchng_reactions
        nametoid[model.name] = nmtoid
        exrn_to_exmet[model.name] = exrxn_to_met
        exmet_to_exrn[model.name] = met_to_exrxn





    ##### NOW: we have to reconcile the exchanged metabolites. Swapping order means swapping rows of GammaStar! So
    ### we must agree on an order.
    masterlist = []
    for li in metaabs.values():
        masterlist += li
    masterlist = np.unique(masterlist)



    #### Initial y is not as difficult. I'll take the highest.
    master_y0 = {}
    for nm in masterlist:
        yyy0 = 0
        # ctt = 0
        for mod in y0s.values():
            if nm in mod.keys():
                yyy0 = max(yyy0,mod[nm])
        master_y0[nm] = yyy0





    surfmods = {}


    for modelkey in models.keys():
        model = models[modelkey]

        gammaStarOrder = np.array([np.where(masterlist == met)[0][0] for met in metaabs[model.name]])

        #Get the stoichiometric matrix and break it apart
        ###Index is metabolite ID, columns are rxn ID
        Gamma = util.array.create_stoichiometric_matrix(model, array_type = 'DataFrame')


        internal_reactions = np.array(Gamma.columns)[[rxn not in exrn[model.name] for rxn in Gamma.columns]]


        internal_metabs = np.array(Gamma.index)[[met not in metabids[model.name] for met in Gamma.index]]

        EyE = Gamma.loc[np.array(metabids[model.name]),np.array(exrn[model.name])]





        if np.all(EyE == -np.eye(EyE.shape[0])):#If -I, then -influx + gammaStar*v = 0, so influx = gammaStar*v...but that appears to be wrong somehow...
            GammaStar = -Gamma.loc[np.array(metabids[model.name]),internal_reactions]
        elif np.all(EyE == np.eye(EyE.shape[0])):#else influx = -gammaStar*v
            GammaStar = Gamma.loc[np.array(metabids[model.name]),internal_reactions]
        else:
            print("[prep_cobrapy_models] ",model.name,": Left of GammaStar is not +/- identity, problem with block form.")
            return None


        # =============================================================================

        GammaDagger = Gamma.loc[internal_metabs,internal_reactions]
        GammaStarar = GammaStar.values
        GammaDaggerar = GammaDagger.values


        #Next we need the objective function that identifies growth - the flux that COBRA optimizes
        growth_col = pd.Series(np.zeros(len(internal_reactions)),index = GammaDagger.columns)
        for rxn in util.solver.linear_reaction_coefficients(model).keys():
            growth_col.loc[rxn.id] = util.solver.linear_reaction_coefficients(model)[rxn]


        lilgamma = growth_col.values
        internal_upper_bounds = np.array([model.reactions.get_by_id(rxnid).upper_bound for rxnid in internal_reactions])
        internal_lower_bounds = np.array([-model.reactions.get_by_id(rxnid).lower_bound for rxnid in internal_reactions])


######## Upper Bounds


        if modelkey in upper_bound_functions.keys():
            if isinstance(upper_bound_functions[modelkey],dict):
                uftype = "User"
                exub = np.array([upper_bound_functions[modelkey][met] if met in upper_bound_functions[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                if modelkey in (upper_bound_functions_dt.keys() and isinstance(upper_bound_functions_dt[modelkey],dict)):
                    exubdt = np.array([upper_bound_functions_dt[modelkey][met] if met in upper_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
            elif upper_bound_functions[modelkey] == "constant":
                uftype = "Constant"
                try:
                    flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake\n".format(modelkey))
                except:
                    print("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake".format(modelkey))
                exub = np.array([lambda x,val=master_y0[metab]: val for metab in metaabs[model.name]])
                exubdt = np.array([lambda x : 0  for i in range(len(metaabs[model.name]))])
            elif upper_bound_functions[modelkey] == "linearRand":
                uftype = "Linear"
                try:
                    flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with random coefficients\n".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Upper Bounds: Using linear uptake with random coefficients")
                rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
                exub = np.array([lambda x,i=i,r = rands[i]: r*x[i] for i in range(len(metaabs[model.name]))])
                exubdt = np.array([lambda x,xd, i=i,r=rands[i] : r*xd[i] for i in range(len(metaabs[model.name]))])
            elif upper_bound_functions[modelkey] == "linearScale":
                uftype = "Linear"
                try:
                    flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with uniform coefficients = {}\n".format(modelkey,linearScale))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Upper Bounds: Using linear uptake with uniform coefficients = {}".format(linearScale))
                exub = np.array([lambda x,i=i: linearScale*x[i] for  i in range(len(metaabs[model.name]))])
                exubdt = np.array([lambda x,xd,i=i: linearScale*xd[i] for  i in range(len(metaabs[model.name]))])
            elif upper_bound_functions[modelkey] == "hill1Rand":
                uftype = "Hill"
                try:
                    flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Upper Bounds: Using hill function uptake with exponent = 1, Kd random")
                rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
                exub = np.array([lambda x,i=i,r=rands[i]: x[i]/(r + x[i]) for i in range(len(metaabs[model.name]))])
                exubdt = np.array([lambda x,xd,i=i,r=rands[i]: xd[i]*r/((r+x[i])**2) for i in range(len(metaabs[model.name]))])
            elif upper_bound_functions[modelkey] == "hill11":
                uftype = "Hill"
                try:
                    flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Upper Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1")
                exub = np.array([lambda x,i=i: x[i]/(1 + x[i]) for i in range(len(metaabs[model.name]))])
                exubdt = np.array([lambda x,xd,i=i: xd[i]*1/((1+x[i])**2) for i in range(len(metaabs[model.name]))])
            else:
                print("[prep_cobrapy_models] Unknown function type {}. Prep CobraPy Model failed.".format(upper_bound_functions[modelkey]))
                return None

        elif ub_funs == "constant":
            uftype = "Constant"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake".format(modelkey))
            exub = np.array([lambda x,val=master_y0[metab]: val for metab in metaabs[model.name]])
            exubdt = np.array([lambda x,xd : 0  for i in range(len(metaabs[model.name]))])
        elif ub_funs == "linearRand":
            uftype = "Linear"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with random coefficients\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with random coefficients".format(modelkey))
            rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
            exub = np.array([lambda x,i=i,r = rands[i]: r*x[i] for i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd, i=i,r=rands[i] : r*xd[i] for i in range(len(metaabs[model.name]))])
        elif ub_funs == "linearScale":
            uftype = "Linear"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with uniform coefficients = {}\n".format(modelkey,linearScale))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake with uniform coefficients = {}".format(modelkey,linearScale))
            exub = np.array([lambda x,i=i: linearScale*x[i] for  i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd,i=i: linearScale*xd[i] for  i in range(len(metaabs[model.name]))])
        elif ub_funs == "hill1Rand":
            uftype = "Hill"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
            rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
            exub = np.array([lambda x,i=i,r=rands[i]: x[i]/(r + x[i]) for i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd,i=i,r=rands[i]: xd[i]*r/((r+x[i])**2) for i in range(len(metaabs[model.name]))])
        elif ub_funs == "hill11":
            uftype = "Hill"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
            exub = np.array([lambda x,i=i: x[i]/(1 + x[i]) for i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd,i=i: xd[i]*1/((1+x[i])**2) for i in range(len(metaabs[model.name]))])
        else:
            print("[prep_cobrapy_models] {} Unknown function type {}. Prep CobraPy Model failed.".format(modelkey,ub_funs))
            return None



######### Lower Bounds

        if modelkey in lower_bound_functions.keys():
            if isinstance(lower_bound_functions[modelkey],dict):
                lftype = "User"
                exlb = np.array([lower_bound_functions[modelkey][met] if met in lower_bound_functions[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                if modelkey in lower_bound_functions_dt.keys():
                    exlbdt = np.array([lower_bound_functions_dt[modelkey][met] if met in lower_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
            elif lower_bound_functions[modelkey] == "constant":
                lftype = "Constant"
                try:
                    flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake\n".format(modelkey))
                except:
                    print("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake".format(modelkey))
                exlb = np.array([lambda x,val = model.reactions.get_by_id(exmet_to_exrn[model.name][metab]).upper_bound : val  for metab in metabids[model.name]])
                exlbdt = np.array([lambda x,xd : 0  for i in range(len(metabids[model.name]))])
            elif lower_bound_functions[modelkey] == "linearRand":
                lftype = "Linear"
                try:
                    flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with random coefficients\n".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Lower Bounds: Using linear uptake with random coefficients")
                rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
                exlb = np.array([lambda x,i=i,r=rands[i] : r*x[i] for i in range(len(metaabs[model.name]))])
                exlbdt = np.array([lambda x,xd,i=i,r=rands[i]: r*xd[i] for i in range(len(metaabs[model.name]))])
            elif lower_bound_functions[modelkey] == "linearScale":
                lftype = "Linear"
                try:
                    flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with uniform coefficients = {}\n".format(modelkey,linearScale))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Lower Bounds: Using linear uptake with uniform coefficients = {}".format(linearScale))                
                exlb = np.array([lambda x,i=i: linearScale*x[i] for  i in range(len(metaabs[model.name]))])
                exlbdt = np.array([lambda x,xd,i=i: linearScale*xd[i] for  i in range(len(metaabs[model.name]))])
            elif lower_bound_functions[modelkey] == "hill1Rand":
                lftype = "Hill"
                try:
                    flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Lower Bounds: Using hill function uptake with exponent = 1, Kd random")
                rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
                exlb = np.array([lambda x,i=i,r=rands[i]: x[i]/(r + x[i]) for i in range(len(metaabs[model.name]))])
                exlbdt = np.array([lambda x,xd,i=i,r=rands[i]: xd[i]*r/((r+x[i])**2) for i in range(len(metaabs[model.name]))])
            elif lower_bound_functions[modelkey] == "hill11":
                lftype = "Hill"
                try:
                    flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
                except:
                    print("[prep_cobrapy_models] ",modelkey, " Lower Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1")
                exlb = np.array([lambda x,i=i: x[i]/(1 + x[i]) for i in range(len(metaabs[model.name]))])
                exlbdt = np.array([lambda x,xd,i=i: xd[i]*1/((1+x[i])**2) for i in range(len(metaabs[model.name]))])
            else:
                print("[prep_cobrapy_models]",model.name," Lower Bounds Unknown function type {}. Prep CobraPy Model failed.".format(lower_bound_functions[modelkey]))
                return None

        elif lb_funs == "constant":
            lftype = "Constant"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake".format(modelkey))
            exlb = np.array([lambda x,val = model.reactions.get_by_id(exmet_to_exrn[model.name][metab]).upper_bound : val  for metab in metabids[model.name]])
            exlbdt = np.array([lambda x,xd : 0  for i in range(len(metabids[model.name]))])
        elif lb_funs == "linearRand":
            lftype = "Linear"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with random coefficients\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with random coefficients".format(modelkey))
            rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
            exlb = np.array([lambda x,i=i,r=rands[i] : r*x[i] for i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,i=i,r=rands[i]: r*xd[i] for i in range(len(metaabs[model.name]))])
        elif lb_funs == "linearScale":
            lftype = "Linear"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with uniform coefficients = {}\n".format(modelkey,linearScale))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake with uniform coefficients = {}".format(modelkey,linearScale))
            exlb = np.array([lambda x,i=i: linearScale*x[i] for  i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,i=i: linearScale*xd[i] for  i in range(len(metaabs[model.name]))])
        elif lb_funs == "hill1Rand":
            lftype = "Hill"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd random".format(modelkey))
            rands = [np.random.rand() for i in range(len(metaabs[model.name]))]
            exlb = np.array([lambda x,i=i,r=rands[i]: x[i]/(r + x[i]) for i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,i=i,r=rands[i]: xd[i]*r/((r+x[i])**2) for i in range(len(metaabs[model.name]))])
        elif lb_funs == "hill11":
            lftype = "Hill"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake with exponent = 1, Kd uniformly = 1".format(modelkey))
            exlb = np.array([lambda x,i=i: x[i]/(1 + x[i]) for i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,i=i: xd[i]*1/((1+x[i])**2) for i in range(len(metaabs[model.name]))])
        else:
            print("[prep_cobrapy_models] {} Lower Bounds Unknown function type {}. Prep CobraPy Model failed.".format(modelkey,lftype))
            return None

        #SurfMod(gamStar,gamDag,objective,intrn_order,exrn_order,interior_lbs,interior_ubs,exterior_lbfuns,exterior_ubfuns,exterior_lbfuns_derivative = [],exterior_ubfuns_derivatives = [],exchanged_metabolites,Name = None,deathrate = 0)
        drt = deathrates.get(modelkey,0)
        surfmods[modelkey] = SurfMod(metaabs[model.name],GammaStarar,GammaDaggerar,lilgamma,internal_reactions,internal_lower_bounds,internal_upper_bounds,exlb,exub,exlbdt,exubdt,deathrate = drt,lbfuntype = lftype,ubfuntype = uftype,Name = model.name, gamma_star_indices = gammaStarOrder, forcedOns=forceOns)


    return surfmods,masterlist,master_y0
