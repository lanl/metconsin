import numpy as np
import scipy as sp
from surfmod import *
import pandas as pd

def prep_cobrapy_models(models,**kwargs):


    """
    
    Creates a set of SurfMod objects from a set of cobrapy GSMs, and interprets the media (either given by the user or inferred from the cobrapy models) to create an initial metbaolite concentration. 

    :param media: Growth media used for the simulation. If None, averages the media dicts packaged with the GSMs used. Can be keyed by metabolite name, ID, or exchange reaction ID. Alternatively, passing the string "minimal" attempts to define a minimal growth media for each microbe and averages that. Default None.
    :type media: dict or str


    :param ub_funs: General function to use for upper bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``.  Default ``linear``
    :type ub_funs: str
    :param lb_funs: General function to use for lower bounds on exchange reactions. Options are ``model``, ``constant``, ``linear``, ``hill``, or ``user``. Default ``model``
    :type lb_funs: str
    :param ub_params: Parameters to use for upper bound functions (see below).
    :type ub_params: dict
    :param lb_params: Parameters to use for lower bound functions (see below).
    :type lb_params: dict

    :param upper_bound_user_functions: User-defined upper bound functions for exchange reactions. **Ignored** unless ``ub_funs`` == ``user``. Any model not included (which can be all models), will default to functions defined by the ``ub_funs`` parameter. Default None.
    :type upper_bound_user_functions: dict[str,array of lambdas]
    :param lower_bound_user_functions: User-defined lower bound functions for exchange reactions. **Ignored** unless ``lb_funs`` == ``user``.Any model not included (which can be all models), will default to functions defined by the ``lb_funs`` parameter. Default None.
    :type lower_bound_user_functions: dict[str,array of lambdas]    
    :param upper_bound_user_functions_dt: Derivatives of user defined upper_bound_functions_dt. Default None
    :type upper_bound_user_functions_dt:  dict[str,array of lambdas]
    :param lower_bound_user_functions_dt: kwargs.get(lower_bound_functions_dt)
    :type lower_bound_user_functions_dt:  dict[str,array of lambdas]

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

    Upper and lower bound functions can be user defined seperately for each bound and each model with a dict (keyed by model name) of arrays of lambda functions. The 
    user must provide the derivative of the functions with respect to time in ``upper_bound_functions_dt`` and/or ``lower_bound_functions_dt``. 
    
    Alternatively, predifined functions are provided. The user may include parameters for the predefined functions with lb_params and ub_params. The predifined options are:

    - *model*\ : Same as constant, but values are taken from bounds defined in the COBRA model file.
    - *constant*\ : Bounds do not depend on the metabolites. params supplied (default 1) give constant uptake bound.
    - *linear*\ : Bounds are linear in the corresponding metabolite with a random constant of proportionality. Provide an array of proportionality rates. Default 1.
    - *hill*\ : Bounds follow a Hill function of the corresponding metabolite. Provide an arrays of tuples with (km,kd,n). Default all 1.

    Alternatively, params for each model can be supplied as a dictionary keyed by metabolite name. In this case, hill params should be a dictionary of tuples

    Hill functions are defined as

    .. math::

        h(y) = k_m\\frac{y^n}{k_d + y^n}


    """


    #can provide metabolite uptake dictionary as dict of dicts {model_key1:{metabolite1:val,metabolite2:val}}





    forceOns = kwargs.get("forceOns",True)

    lb_funs = kwargs.get("lb_funs","model")############################
    ub_funs = kwargs.get("ub_funs","linear")######################
    lb_params = kwargs.get("lb_params",{})################################
    ub_params = kwargs.get("ub_params",{})################################

    upper_bound_functions = kwargs.get("upper_bound_user_functions",{})#####################
    lower_bound_functions = kwargs.get("lower_bound_user_functions",{})####################
    upper_bound_functions_dt = kwargs.get("upper_bound_user_functions_dt",{})##################
    lower_bound_functions_dt = kwargs.get("lower_bound_user_functions_dt",{})###################

    media = kwargs.get("media",{})
    met_filter = kwargs.get("met_filter",[])
    met_filter_sense = kwargs.get("met_filter_sense","exclude")

    flobj = kwargs.get("flobj")
    deathrates = kwargs.get("deathrates")


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


        if ub_funs == "user":
            uftype = "User"
            if isinstance(upper_bound_functions[modelkey],dict):
                exub = np.array([upper_bound_functions[modelkey][met] if met in upper_bound_functions[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                if modelkey in (upper_bound_functions_dt.keys() and isinstance(upper_bound_functions_dt[modelkey],dict)):
                    exubdt = np.array([upper_bound_functions_dt[modelkey][met] if met in upper_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
            else:
                exub = upper_bound_functions[modelkey]
                if modelkey in upper_bound_functions_dt.keys():
                    if isinstance(upper_bound_functions_dt[modelkey],dict):
                        exubdt = np.array([upper_bound_functions_dt[modelkey][met] if met in upper_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                    else:
                        exubdt = upper_bound_functions_dt[modelkey]
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
                
        elif ub_funs == "model":
            uftype = "Constant"
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake equal to initial metabolite concentration\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake equal to initial metabolite concentration".format(modelkey))
            exub = np.array([lambda x,val=master_y0[metab]: val for metab in metaabs[model.name]])
            exubdt = np.array([lambda x : 0  for i in range(len(metaabs[model.name]))])
        elif ub_funs == "constant":
            uftype = "Constant"
            ubpar = ub_params.get(modelkey,np.ones(len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using constant uptake".format(modelkey))
            if isinstance(ubpar,dict):
                exub = np.array([lambda x,val=ubpar[metab]: val for metab in metaabs[model.name]])
                exubdt = np.array([lambda x,xd : 0  for i in range(len(metaabs[model.name]))])
            else:
                exub = np.array([lambda x,val=ubpar[i]: val for i in range(len(metaabs[model.name]))])
                exubdt = np.array([lambda x,xd : 0  for i in range(len(metaabs[model.name]))])

        elif ub_funs == "linear":
            uftype = "Linear"
            ubpar = ub_params.get(modelkey,np.ones(len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using linear uptake".format(modelkey))
            if isinstance(ubpar,dict):

                ubpar = [ubpar[me] for me in metaabs[model.name]]

            exub = np.array([lambda x,j=i,val=ubpar[i]: val*x[j] for i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd,j=i,val=ubpar[i]: val*xd[j] for i in range(len(metaabs[model.name]))])
                
        elif ub_funs == "hill":
            uftype = "Linear"
            ubpar = ub_params.get(modelkey,np.array([(1,1,1)]*len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Upper Bounds: Using hill function uptake".format(modelkey))
            if isinstance(ubpar,dict):

                ubpar = [ubpar[me] for me in metaabs[model.name]]

            exub = np.array([lambda x,j=i,km=ubpar[i][0],kd=ubpar[i][1],n=ubpar[i][2]: km*x[j]**n/(kd+x[j]**n) for i in range(len(metaabs[model.name]))])
            exubdt = np.array([lambda x,xd,j=i,km=ubpar[i][0],kd=ubpar[i][1],n=ubpar[i][2]: xd[j]*(n*kd*km*x[j]**(n-1))/((kd+x[j]**n)**2) for i in range(len(metaabs[model.name]))])
        
        else:
            try:
                flobj.write("[prep_cobrapy_models] {} Please choose 'constant', 'linear', 'hill', or 'user' for upper bound functions.\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Please choose 'constant', 'linear', 'hill', or 'user' for upper bound functions.".format(modelkey))



######### Lower Bounds

        if lb_funs == "user":
            lftype = "User"
            if isinstance(lower_bound_functions[modelkey],dict):
                exlb = np.array([lower_bound_functions[modelkey][met] if met in lower_bound_functions[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                if modelkey in (lower_bound_functions_dt.keys() and isinstance(lower_bound_functions_dt[modelkey],dict)):
                    exlbdt = np.array([lower_bound_functions_dt[modelkey][met] if met in lower_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
            else:
                exlb = lower_bound_functions[modelkey]
                if modelkey in lower_bound_functions_dt.keys():
                    if isinstance(lower_bound_functions_dt[modelkey],dict):
                        exlbdt = np.array([lower_bound_functions_dt[modelkey][met] if met in lower_bound_functions_dt[modelkey].keys() else lambda x : 0 for met in metaabs[model.name]])
                    else:
                        exlbdt = lower_bound_functions_dt[modelkey]
                else:
                    print("[prep_cobrapy_models] Error: Numeric differentiation not supported. Please provide derivative for user-defined bound functions")
                    return None
        elif lb_funs == "model":
            lftype = "Constant"
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake as defined in COBRA model file\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake as defined in COBRA model file".format(modelkey))
            exlb = np.array([lambda x,val = model.reactions.get_by_id(exmet_to_exrn[model.name][metab]).upper_bound : val  for metab in metabids[model.name]])
            exlbdt = np.array([lambda x,xd : 0  for i in range(len(metabids[model.name]))])
        elif lb_funs == "constant":
            lftype = "Constant"
            lbpar = lb_params.get(modelkey,np.ones(len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using constant uptake".format(modelkey))
            if isinstance(lbpar,dict):
                exlb = np.array([lambda x,val=lbpar[metab]: val for metab in metaabs[model.name]])
                exlbdt = np.array([lambda x,xd : 0  for i in range(len(metaabs[model.name]))])
            else:
                exlb = np.array([lambda x,val=lbpar[i]: val for i in range(len(metaabs[model.name]))])
                exlbdt = np.array([lambda x,xd : 0  for i in range(len(metaabs[model.name]))])

        elif lb_funs == "linear":
            lftype = "Linear"
            lbpar = lb_params.get(modelkey,np.ones(len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using linear uptake".format(modelkey))
            if isinstance(lbpar,dict):

                lbpar = [lbpar[me] for me in metaabs[model.name]]

            exlb = np.array([lambda x,j=i,val=lbpar[i]: val*x[j] for i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,j=i,val=lbpar[i]: val*xd[j] for i in range(len(metaabs[model.name]))])
                
        elif lb_funs == "hill":
            lftype = "Linear"
            lbpar = lb_params.get(modelkey,np.array([(1,1,1)]*len(metaabs[model.name])))
            try:
                flobj.write("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Lower Bounds: Using hill function uptake".format(modelkey))
            if isinstance(lbpar,dict):

                lbpar = [lbpar[me] for me in metaabs[model.name]]

            exlb = np.array([lambda x,j=i,km=lbpar[i][0],kd=lbpar[i][1],n=lbpar[i][2]: km*x[j]**n/(kd+x[j]**n) for i in range(len(metaabs[model.name]))])
            exlbdt = np.array([lambda x,xd,j=i,km=lbpar[i][0],kd=lbpar[i][1],n=lbpar[i][2]: xd[j]*(n*kd*km*x[j]**(n-1))/((kd+x[j]**n)**2) for i in range(len(metaabs[model.name]))])
        else:
            try:
                flobj.write("[prep_cobrapy_models] {} Please choose 'constant', 'linear', 'hill', or 'user' for upper bound functions.\n".format(modelkey))
            except:
                print("[prep_cobrapy_models] {} Please choose 'constant', 'linear', 'hill', or 'user' for upper bound functions.".format(modelkey))

        #SurfMod(gamStar,gamDag,objective,intrn_order,exrn_order,interior_lbs,interior_ubs,exterior_lbfuns,exterior_ubfuns,exterior_lbfuns_derivative = [],exterior_ubfuns_derivatives = [],exchanged_metabolites,Name = None,deathrate = 0)
        drt = deathrates.get(modelkey,0)
        surfmods[modelkey] = SurfMod(metaabs[model.name],GammaStarar,GammaDaggerar,lilgamma,internal_reactions,internal_lower_bounds,internal_upper_bounds,exlb,exub,exlbdt,exubdt,deathrate = drt,lbfuntype = lftype,ubfuntype = uftype,Name = model.name, gamma_star_indices = gammaStarOrder, forcedOns=forceOns)


    return surfmods,masterlist,master_y0
