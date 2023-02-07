import surfmod as sm
import prep_models as pr
import make_network as mn
import time
import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import root_scalar


def evolve_community(t,s,models,metabolite_in,metabolite_out):
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
    y = s[len(models):]
    v = np.array([])
    for i in range(len(models)):
        model = models[i]
        model.compute_internal_flux(y)
        model.compute_slacks(y)
        v = np.append(v,np.concatenate([model.inter_flux,model.slack_vals]))#[model.current_basis[2]]
    return min(v) + 10**-8

def get_var_i(t,i,sln,mod,nummods):
    y = sln(t)[nummods:]
    mod.compute_internal_flux(y)
    mod.compute_slacks(y)
    return np.concatenate([mod.inter_flux,mod.slack_vals])[i]

def find_stop(t0,t1,sln,models):
    if t0==t1:
        print("[find_stop] No Interval")
        return t0
    all_roots = t1*np.ones(sum([mod.total_vars for mod in models]))
    j = 0
    for mod in models:
        for i in range(mod.total_vars):
            if min(abs(get_var_i(t0,i,sln,mod,len(models))),abs(get_var_i(t1,i,sln,mod,len(models)))) > 10**-7 and (get_var_i(t0,i,sln,mod,len(models))*get_var_i(t1,i,sln,mod,len(models)) < 0):
                all_roots[j] = root_scalar(get_var_i,args = (i,sln,mod,len(models)),bracket=[t0,t1]).root
            j+=1
    return min(all_roots)


def surfin_fba(models,x0,y0,endtime, 
                solver = 'gurobi',
                track_fluxes = True, 
                save_bases = True,
                save_internal_flux = True, 
                resolution = 0.1,
                report_activity = True, 
                flobj = None,
                fwreport = False,
                debugging = False,
                inflow = None,
                outflow = None
                ):


    '''
    models = list of models.
    '''
    


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
            stptime = find_stop(t_c, interval.t[-1],interval.sol,models)#interval.t[-1]#find_event(interval,models)
        else:
            stptime = t_c

        
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
        retdict = {"t":np.concatenate(t),"x":np.concatenate(x,axis = 1),"y":np.concatenate(y,axis = 1),"bt":np.array(basis_change)}
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
