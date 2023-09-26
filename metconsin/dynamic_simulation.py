# from . import surfmod as sm
# from . import prep_models as pr
# from . import make_network as mn
import time
import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import root_scalar


def evolve_community(t,s,models,metabolite_in,metabolite_out):

    """

    ODE right hand side to be integrated during intervals of smoothness in dynamic FBA.

    :param t: time-point in simulation
    :type t: float
    :param s: current value of the state vector of the simulation (biomass of all taxa and metabolites)
    :type s: array[float]
    :param models: GSMs used in simulation
    :type models: list[SurfMod]
    :param yi: metabolite inflow 
    :type yi: array[float]
    :param yo: metabolite outflow 
    :type yo: array[float]

    :return: value of the vector field at the current time-point and system state
    :rtype: array[float]

    """


    x = s[:len(models)]
    y = s[len(models):]
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    for i in range(len(models)):
        model = models[i]
        model.compute_internal_flux(y)
        xdot[i] = x[i]*np.dot(-model.objective,model.inter_flux) - x[i]*model.deathrate
        ydi = np.zeros_like(ydot)
        ydi[model.ExchangeOrder] = -x[i]*np.dot(model.GammaStar,model.inter_flux)#Influx = GammaStar*v
        ydot = ydot + ydi
    
    ydot = ydot + metabolite_in - y*metabolite_out

    return np.concatenate([xdot,ydot])

def all_vs(t,s,models,yi,yo):

    """

    Computes a flattened array of all the fluxes of all the models so that ``scipy.integrate.solve_ivp`` can track for a 
    stopping condition on integration (infeasibility of one of the fluxes)

    :param t: time-point in simulation
    :type t: float
    :param s: current value of the state vector of the simulation (biomass of all taxa and metabolites)
    :type s: array[float]
    :param models: GSMs used in simulation
    :type models: list[SurfMod]
    :param yi: metabolite inflow (unused but this function must have the same parameters as the function being integrated)
    :type yi: array[float]
    :param yo: metabolite outflow (unused but this function must have the same parameters as the function being integrated)
    :type yo: array[float]

    :return: minimum flux across all models, with a small perturbation added so that the integrator only stops if this minimum 
    becomes strictly negative.
    :rtype: float

    **Modifies** 
    
    - ``model.inter_flux`` for each model (see :py:func:`model.compute_internal_flux <surfmod.SurfMod.compute_internal_flux>`)
    - ``model.slack_vals`` for each model (see :py:func:`model.compute_slacks <surfmod.SurfMod.compute_slacks>`)

    """

    y = s[len(models):]
    v = np.array([])
    for i in range(len(models)):
        model = models[i]
        model.compute_internal_flux(y)
        model.compute_slacks(y)
        v = np.append(v,np.concatenate([model.inter_flux,model.slack_vals]))#[model.current_basis[2]]
    return min(v) + 10**-8

def get_mod_v(s,model,nummods):

    '''
    returns vector of all internal fluxes for a model (including slack values) in order to find infeasibility as a stopping condition on integration

    :param s: current value of the state vector of the simulation (biomass of all taxa and metabolites)
    :type s: array[float]
    :param model: GSM used in the simulation
    :type model: SurfMod

    :return: internal flux vector as [model.interflux,model.slack_vals]
    :rtype: array[float]

    **Modifies** 
    
    - ``model.inter_flux`` for each model (see :py:func:`model.compute_internal_flux <surfmod.SurfMod.compute_internal_flux>`)
    - ``model.slack_vals`` for each model (see :py:func:`model.compute_slacks <surfmod.SurfMod.compute_slacks>`)

    '''
    y = s[nummods:]
    model.compute_internal_flux(y)
    model.compute_slacks(y)
    v = np.concatenate([model.inter_flux,model.slack_vals])
    return v

def get_var_i(t,i,sln,mod,nummods):

    """
    
    CURRENTLY UNUSED

    Function to get fluxes so that they can be checked for feasibility in :py:func:`find_stop <dynamic_simulation.find_stop>` 

    :param t: time-point in the simulation at which to compute a flux
    :type t: float
    :param mod: GSM for which to compute a flux
    :type mod: SurfMod
    :param i: index of a flux in the GSM
    :type i: int
    :param sln: ODE solution including the time-point t
    :type sln: solve_ivp solution object
    :param nummods: The number of taxa in the community
    :type nummods: int

    :return: flux value of flux i at time t for model mod
    :rtype: float

    """

    y = sln(t)[nummods:]
    mod.compute_internal_flux(y)
    mod.compute_slacks(y)
    return np.concatenate([mod.inter_flux,mod.slack_vals])[i]

def find_stop(t0,t1,sln,models):
    """
    Uses ``scipy.optimize.root_scalar`` to refine the estimate of when a flux because infeasible and a new basis was needed within
    an interval of smooth forward ODE solving. If the interval is legnth 0, a warning is printed and the stop time is returned as the
    start of the interval.

    :param t0: lower bound of time interval
    :type t0: float
    :param t1: upper bound of time interval
    :type t1: float
    :param sln: ODE solution to dynamic FBA within the interval
    :type sln: solve_ivp solution object
    :param models: GSMs used in the simulation
    :type models: list[SurfMod]
    :return: minimum time at which any model flux became infeasible.
    :rtype: float

    """


    if t0==t1:
        print("[find_stop] No Interval")
        return t0
    all_roots = t1*np.ones(sum([mod.total_vars for mod in models]))
    j = 0
    for mod in models:
        for i in range(mod.total_vars):
            if min(abs(get_var_i(t0,i,sln,mod,len(models))),abs(get_var_i(t1,i,sln,mod,len(models)))) > 10**-8 and (get_var_i(t0,i,sln,mod,len(models))*get_var_i(t1,i,sln,mod,len(models)) < 0):
                all_roots[j] = root_scalar(get_var_i,args = (i,sln,mod,len(models)),bracket=[t0,t1]).root
                print("AN EARLIER STOP: {} - {}".format(t1,all_roots[j]))
            j+=1
    return min(all_roots)


def surfin_fba(models,x0,y0,endtime,**kwargs):


    '''
    
    Runs the surfin_fba method to generate dymamic FBA simulations, saving bases used for constructing networks. Initializes the simulation
    using flux balance analysis (with the method :py:func:`fba_gb <surfmod.SurfMod.fba_gb>` or :py:func:`fba_clp <surfmod.SurfMod.fba_clo>`) 
    and the :py:func:`findwaves <surfmod.SurfMod.findwaves>` method of the SurfMod class and simulates forward until a flux is no longer
    feasible. It then reuses :py:func:`findwaves <surfmod.SurfMod.findwaves>` to find a new basis for the model with an infeasible flux. 
    It also attempts to find new bases for all the other models, in case there is a basis which will last longer before infeasibility. 
    Tracks which bases changed and saves the indices of the old bases for network building, if that option is True. Infeasibility of a flux 
    is first detected by the ODE solver, using the ``events`` option for ``scipy.integrate.solve_ivp``, and then refined using 
    :py:func:`find_stop <dynamic_simulation.find_stop>`. If the solutions reaches the prescribed end-time or cannot find new bases
    for simulation, it stops.

    :param models: list of GSMs used in simulation, as SurfMod objects
    :type community_members: list[SurfMod]
    :param x0: Starting community abundances (treated as absolute abundances.) Array with same length as models, in order of models.
    :type x0: numpy array
    :param y0: Starting metabolite concentrations. Size of array depends on models.
    :type y0: numpy array
    :param endtime: Simulation length. Default 10**-2
    :type endtime: float
    :param resolution: Time-resolution of the dynamics output. Default 0.1
    :type resolution: float
    :param inflow: Inflow rate for each metabolite. Default all 0
    :type inflow: array[float]
    :param outflow: Outflow rate for each metabolite. Default all 0
    :type outflow: array[float]
    :param solver: LP solver to use (currently supports ``gurobi`` and ``clp``). Default ``gurobi``
    :type solver: str
    :param save_bases: Whether or not to save information about the bases used in simulation. Default True
    :type save_bases: bool
    :param track_fluxes: Whether or not to save the exchange fluxes computed during the simulation. Default True
    :type track_fluxes: bool
    :param save_internal_flux: Whether or not to save the internal fluxes computed during the simulation. Default True
    :type save_internal_flux: bool
    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File
    :param report_activity: Whether or not to log simulation progress. Default True
    :type report_activity: bool
    :param fwreport: Whether or not to log details of basis finding. Default False
    :type fwreport: bool
    :param debugging: Turn on some debugging prints. Default False
    :type debugging: bool
    :param refine_intervals: whether to look for interval endtimes earlier than provided by solve_ivp "events". Default False
    :type refine_intervals: bool

    
    :return: Dictionary containing the simulation. Keys are :

    - *t*\ : timepoints for the simulation. Numpy array length T
    - *x*\ : dynamics of the microbial taxa. Numpy array shape NumTaxa x T 
    - *y*\ : dynamics of the metabolites. Numpy array shape NumMetabolites x T
    - *bt*\ : list of times a basis was recomputed.
    - *bf*\ : dict indicating which flux of which model triggered the basis change.
    - *basis* (if ``save_basis``): bases used
    - *Exchflux* (if ``track_fluxes``): Exchange fluxes - dict of arrays
    - *Intflux* (if ``save_internal_flux``): Internal fluxes - dict of arrays

    :rtype: dict 

    '''
    

    solver = kwargs.get("solver",'gurobi')
    track_fluxes = kwargs.get("track_fluxes",True) 
    save_bases = kwargs.get("save_bases",True)
    save_internal_flux = kwargs.get("save_internal_flux",True)
    resolution = kwargs.get("resolution",0.1)
    report_activity = kwargs.get("report_activity",True)
    flobj = kwargs.get("flobj")
    fwreport = kwargs.get("fwreport",False)
    debugging = kwargs.get("debugging",False)
    inflow = kwargs.get("inflow",None)
    outflow = kwargs.get("outflow",None)
    refine_stoptime = kwargs.get("refine_intervals",False)

    t1 = time.time()
    if report_activity:
        try:
            flobj.write("surfin_fba: Initializing Simulation\n")
        except:
            print("surfin_fba: Initializing Simulation")
    
    try:
        if len(inflow) != len(y0):
            inflow = np.zeros_like(y0)
    except:
        if inflow == None:
            inflow = np.zeros_like(y0)
    try:
        if len(outflow) != len(y0):
            outflow = np.zeros_like(y0)
    except:
        if outflow == None:
            outflow = np.zeros_like(y0)

    # model_names = [model.Name for model in models]
    s0 = np.concatenate([x0,y0])
    ydot0 = np.zeros_like(y0)
    fluxes = {}
    for i in range(len(models)):#model in models:
        #fba_solver sets model.essential_basis,model.current_basis_full,model.inter_flux
        model = models[i]
        if solver == 'gurobi':
            obval = model.fba_gb(y0,flobj = flobj)#,secondobj = None)
        elif solver == 'clp':
            obval = model.fba_clp(y0,flobj = flobj)
        if report_activity:
            try:
                flobj.write("{} Surfin initial growth rate: {}".format(model.Name,obval))
            except:
                print(model.Name, " Surfin initial growth rate: ",obval)
        fluxes[model.Name] = model.inter_flux
        ydi = np.zeros_like(ydot0)
        ydi[model.ExchangeOrder] = -x0[i]*np.dot(model.GammaStar,model.inter_flux)
        ydot0 += ydi
    
    ydot0 = ydot0 + inflow - y0*outflow

    for model in models:
        
        #findWave sets model.current_basis_full, model.current_basis
        model.findWave(y0,ydot0,details = fwreport,flobj = flobj)

        if debugging:
            print("------------------------------------\n\n     Debug  \n\n-----------------------------------------")

            metabolite_con = y0[model.ExchangeOrder]
            exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
            bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

            incess = np.all([j in model.current_basis_full for j in model.essential_basis])
            print("Includes the essentials? {}".format(incess))
            ###
            rk = np.linalg.matrix_rank(model.standard_form_constraint_matrix[:,model.current_basis_full])
            print("Is full rank? {}".format(rk == model.standard_form_constraint_matrix.shape[0]))

            basisflxesbeta = np.linalg.solve(model.standard_form_constraint_matrix[:,model.current_basis_full],bound_rhs)
            basisflxes = np.zeros(model.standard_form_constraint_matrix.shape[1])
            basisflxes[model.current_basis_full] = basisflxesbeta
            fluxesfound = basisflxes[:model.num_fluxes]
            basisval = np.dot(fluxesfound,-model.objective)
            print("Basis gives objective value {}".format(basisval))
            dist = np.linalg.norm(fluxesfound - model.inter_flux)
            print("Distance from basis flux to gurobi flux = {}".format(dist))

            ###Let's check the reduced as well.
            model.compute_internal_flux(y0)
            print("Reduction objective value: {}".format(-np.dot(model.inter_flux,model.objective)))
            print("Reduction error {}".format(np.linalg.norm(model.inter_flux-fluxesfound)))

            neg_ind = np.where(basisflxes<-10**-5)
            if len(neg_ind[0]):
                for ind in neg_ind[0]:
                    print("Negative flux from Waves-Basis at index {} = {}".format(ind,basisflxes[ind]))
            else:
                print("No negative flux from Waves-Basis.")

            print("------------------------------------\n\n      Stop Debug  \n\n-----------------------------------------")


    if save_bases:
        bases = dict([(model.Name,[]) for model in models])
        if all([mod.feasible for mod in models]):
            for model in models:
                bases[model.Name] += [(0,model.current_basis[2])]

    if report_activity:
        if all([mod.feasible for mod in models]):
            bInitial = evolve_community(0,s0,models,inflow,outflow)
            try:
                flobj.write("{} Bases intial growth rate : {}".format([model.Name for model in models],bInitial[:len(models)]/np.array(x0)))
            except:
                print([model.Name for model in models], " Bases intial growth rate :", bInitial[:len(models)]/np.array(x0))
        else:
            try:
                flobj.write("{} Initial feasible? : {}".format([model.Name for model in models],[mod.feasible for mod in models]))
            except:
                print([model.Name for model in models], " Initial feasible? :", [mod.feasible for mod in models])




    t = []
    x = []
    y = []
    if track_fluxes:
        Exchfluxes = dict([(model.Name,[]) for model in models])
    if save_internal_flux:
        intfluxes = dict([(model.Name,[]) for model in models])
    t_c = 0
    basis_change = [t_c]
    basis_reason = {}

    catch_badLP = False
    stops = 0

    while t_c < endtime:
        for model in models:
            if model.current_basis == None:
                catch_badLP = True
        if catch_badLP:
            break

        all_vs.terminal = True
        # all_vs.direction = -1
        # stp_events = [lambda t,s,mods,yi,yo,mod = mod,i=i : get_mod_v(s,mod,len(models))[i]+10**-8 for mod in models for i in range(mod.total_vars)]
        # for ev in stp_events:
        #     ev.terminal = True
        if all([mod.feasible for mod in models]):
            ode_solver = "Radau"
            if report_activity:
                try:
                    flobj.write("surfin_fba: Solving IVP using {} solver\n".format(ode_solver))
                except:
                    print("surfin_fba: Solving IVP using {} solver".format(ode_solver))
            interval = solve_ivp(evolve_community, (t_c,endtime), s0, args=(models,inflow,outflow),events=all_vs,dense_output = True, method=ode_solver)
            if interval.status == -1:
                break
            stptime = interval.t[-1]#find_stop(t_c, interval.t[-1],interval.sol,models)##find_event(interval,models)
            ### If we're worried that solve_ivp's root finder isn't good enough, we can use find_stop to get even more precise
            if refine_stoptime:
                stptime = find_stop(t_c, interval.t[-1],interval.sol,models)
            
            stpflux = []
            for mod in models:
                whr_viol = np.where(get_mod_v(interval.y[:,-1],mod,len(models))+10**-8 < 0)[0]
                if len(whr_viol):
                    whc_viol_v = [j for j in whr_viol if j<len(mod.flux_order)]
                    whc_viol_c = [j-len(mod.flux_order) for j in whr_viol if j>=len(mod.flux_order)]
                    stpflux += [(mod.Name,whc_viol_v,whc_viol_c)]
        else:
            stptime = t_c
            stpflux = None

        
        if stptime == t_c:
            
            ################# IF NOT GOING ANYWHERE###############################
            ########################################################################################################################
            ########################################################################################################################
            ########################################################################################################################
            ########################################################################################################################

            if stops == 0:
                print("No progress at time {}. Relaxing forced on constraints.".format(stptime))
                for model in models:
                    if len(model.ForcedOn):
                        print("Relaxing forced on constraints in model: {}.".format(model.Name))
                        for fon in model.ForcedOn:
                            print("Previously Forced To {}".format(-model.internal_bounds[fon[1]-2*model.num_exch_rxns]))
                            model.internal_bounds[fon[1]-2*model.num_exch_rxns] = 0
                        if solver == 'gurobi':
                            obval = model.fba_gb(s0[len(models):],flobj = flobj)#,secondobj = None)
                        elif solver == 'clp':
                            obval = model.fba_clp(s0[len(models):],flobj = flobj)
                stops += 1


            else:
                print("No progress at time {}. Stopping Simulation.".format(stptime))
                break
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################

        s0 = interval.sol(stptime)
        y0 = np.maximum(0,s0[len(models):])
        T = np.linspace(t_c,stptime,max(2,int((stptime-t_c)*(1/resolution))))
        s = interval.sol(T)
        x += [s[:len(models),:]]
        y += [np.maximum(0,s[len(models):,:])]
        if track_fluxes:
            for model in models:
                for i in range(y[-1].shape[1]):
                    flxt = np.zeros_like(ydot0)
                    model.compute_internal_flux(y[-1][:,i])
                    flxt[model.ExchangeOrder] = np.dot(model.GammaStar,model.inter_flux)
                    Exchfluxes[model.Name] += [flxt]
        if save_internal_flux:
            for model in models:
                for i in range(y[-1].shape[1]):
                    model.compute_internal_flux(y[-1][:,i])
                    intfluxes[model.Name] += [model.inter_flux]
        t += [T]
        t_c = stptime
        if t_c < endtime:
            if report_activity:
                try:
                    flobj.write("surfin_fba: Finding New Basis at time {} \n".format(t_c))
                except:
                    print("surfin_fba: Finding New Basis at time ",t_c)
            yd = evolve_community(t_c,s0,models,inflow,outflow)[len(models):]
            basis_change += [t_c]
            basis_reason[t_c] = stpflux
            for model in models:


                metabolite_con = y[-1][:,-1][model.ExchangeOrder]
                
                exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])


                model.compute_internal_flux(y[-1][:,-1])
                model.compute_slacks(y[-1][:,-1])#slkvals = bound_rhsarr - np.dot(model.solver_constraint_matrix,model.inter_flux)

                all_vars = np.concatenate([model.inter_flux,model.slack_vals])

                model.essential_basis = (all_vars>model.ezero).nonzero()[0]


                updateflg = model.findWave(y[-1][:,-1],yd,details = fwreport,flobj = flobj)
                if save_bases:
                    if updateflg:
                        bases[model.Name] += [(t_c,model.current_basis[2])]

                if debugging:
                    print("------------------------------------\n\n     Debug  \n\n-----------------------------------------")


                    metabolite_con = y[-1][:,-1][model.ExchangeOrder]
                    exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
                    bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

                    incess = np.all([j in model.current_basis_full for j in model.essential_basis])
                    print("Includes the essentials? {}".format(incess))
                    if not incess:
                        for j in model.essential_basis:
                            if j not in model.current_basis_full:
                                print("Missing essential index {} with var value {}".format(j,all_vars[j]))
                    ###
                    rk = np.linalg.matrix_rank(model.standard_form_constraint_matrix[:,model.current_basis_full])
                    print("Is full rank? {}".format(rk == model.standard_form_constraint_matrix.shape[0]))

                    basisflxesbeta = np.linalg.solve(model.standard_form_constraint_matrix[:,model.current_basis_full],bound_rhs)
                    basisflxes = np.zeros(model.standard_form_constraint_matrix.shape[1])
                    basisflxes[model.current_basis_full] = basisflxesbeta
                    fluxesfound = basisflxes[:model.num_fluxes]
                    basisval = np.dot(fluxesfound,-model.objective)
                    print("Basis gives objective value {}".format(basisval))
                    dist = np.linalg.norm(basisflxes - all_vars)
                    print("Distance from basis flux to previous flux = {}".format(dist))

                    ###Let's check the reduced as well.
                    model.compute_internal_flux(y[-1][:,-1])
                    print("Reduction objective value: {}".format(-np.dot(model.inter_flux,model.objective)))
                    print("Reduction error {}".format(np.linalg.norm(model.inter_flux-fluxesfound)))

                    neg_ind = np.where(basisflxes<-10**-5)
                    if len(neg_ind[0]):
                        for ind in neg_ind[0]:
                            print("Negative flux from Waves-Basis at index {} = {}".format(ind,basisflxes[ind]))
                    else:
                        print("No negative flux from Waves-Basis.")

                    print("------------------------------------\n\n      Stop Debug  \n\n-----------------------------------------")


    try:
        retdict = {"t":np.concatenate(t),"x":np.concatenate(x,axis = 1),"y":np.concatenate(y,axis = 1),"bt":np.array(basis_change),"bf":basis_reason}
    except:
        retdict = {"t":[0],"x":np.array([x0]).T,"y":np.array([y0]).T,"bt":[]}
    if track_fluxes:
        retdict["Exchflux"] = {}
        for ky,val in Exchfluxes.items():
            retdict["Exchflux"][ky] = np.array(val).T

    if save_bases:
        retdict["bases"] = bases

    if save_internal_flux:
        retdict["Intflux"] = {}
        for ky,val in intfluxes.items():
            retdict["Intflux"][ky] = np.array(val).T

    if report_activity:
        minuts,sec = divmod(time.time() - t1, 60)
        try:
            flobj.write("surfin_fba: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
        except:
            print("surfin_fba: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.")



    return retdict
