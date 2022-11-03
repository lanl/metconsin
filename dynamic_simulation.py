import surfmod as sm
import prep_models as pr
import make_network as mn
import time
import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import root_scalar


def evolve_community(t,s,models):
    x = s[:len(models)]
    y = s[len(models):]
    xdot = np.zeros_like(x)
    ydot = np.zeros_like(y)

    for i in range(len(models)):
        model = models[i]
        model.compute_internal_flux(y)
        xdot[i] = x[i]*np.dot(-model.objective,model.inter_flux) - model.deathrate
        ydi = np.zeros_like(ydot)
        ydi[model.ExchangeOrder] = -x[i]*np.dot(model.GammaStar,model.inter_flux)#Influx = GammaStar*v
        ydot = ydot + ydi

    return np.concatenate([xdot,ydot])

def all_vs(t,s,models):
    y = s[len(models):]
    v = np.array([])
    for i in range(len(models)):
        model = models[i]
        model.compute_internal_flux(y)
        model.compute_slacks(y)
        v = np.append(v,np.concatenate([model.inter_flux,model.slack_vals]))#[model.current_basis[2]]
    return min(v) + 10**-5

def find_stop(t0,t1,sln,models,cdt = 0.1,fdt=0.01):
    if t0==t1:
        print("[find_stop] No Interval")
        return t0
    cdt = min(cdt,t1-t0)
    fdt = min(fdt,t1-t0)
    #check on a coarse grid for negatives:
    tcrs = np.arange(t0,t1,cdt)
    minpert = np.array([all_vs(t,sln(t),models) for t in tcrs])
    if minpert.round(5).min() >= 0:
        return t1
    else:
        #find first negative
        failat = np.where(minpert.round(5) < 0)[0][0]
        #go finer.
        if failat == 0:
            print("[find_stop] Failed Immediately, value {}".format(minpert[0]))
            return t0
        else:
            tr = np.arange(tcrs[failat-1],tcrs[failat],fdt)
            finerfail = np.array([all_vs(t,sln(t),models) for t in tr])
            flt = tr[np.where(finerfail.round(5) < 0)]
            return flt[0]

def surfin_fba(models,x0,y0,endtime, 
                solver = 'gurobi',
                track_fluxes = True, 
                save_bases = True,
                save_internal_flux = True, 
                resolution = 0.1,
                report_activity = True, 
                flobj = None,
                fwreport = False,
                debugging = False):


    '''
    models = list of models.
    '''
    



    t1 = time.time()
    if report_activity:
        try:
            flobj.write("surfin_fba: Initializing Simulation\n")
        except:
            print("surfin_fba: Initializing Simulation")

    # model_names = [model.Name for model in models]
    s0 = np.concatenate([x0,y0])
    ydot0 = np.zeros_like(y0)
    fluxes = {}
    for i in range(len(models)):#model in models:
        #fba_solver sets model.essential_basis,model.current_basis_full,model.inter_flux
        model = models[i]
        if solver == 'gurobi':
            obval = model.fba_gb(y0)#,secondobj = None)
        elif solver == 'clp':
            obval = model.fba_clp(y0)
        if report_activity:
            try:
                flobj.write(model.Name, " Surfin initial growth rate: ",obval)
            except:
                print(model.Name, " Surfin initial growth rate: ",obval)
        fluxes[model.Name] = model.inter_flux
        ydi = np.zeros_like(ydot0)
        ydi[model.ExchangeOrder] = -x0[i]*np.dot(model.GammaStar,model.inter_flux)
        ydot0 += ydi

    for model in models:
        
        #findWave sets model.current_basis_full, model.current_basis
        model.findWave(y0,ydot0,details = fwreport)

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
        for model in models:
            bases[model.Name] += [model.current_basis]

    if report_activity:
        bInitial = evolve_community(0,s0,models)
        try:
            flobj.write([model.Name for model in models], " Bases intial growth rate :", bInitial[:len(models)]/np.array(x0))
        except:
            print([model.Name for model in models], " Bases intial growth rate :", bInitial[:len(models)]/np.array(x0))




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
        if report_activity:
            try:
                flobj.write("surfin_fba: Solving IVP\n")
            except:
                print("surfin_fba: Solving IVP")
        all_vs.terminal = True
        # all_vs.direction = -1
        if all([mod.feasible for mod in models]):
            interval = solve_ivp(evolve_community, (t_c,endtime), s0, args=(models,), method='RK45',events=all_vs,dense_output = True)
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
                            obval = model.fba_gb(y0)#,secondobj = None)
                        elif solver == 'clp':
                            obval = model.fba_clp(y0)
                stops += 1

            # print("Computed to time {}".format(stptime))



            # try:
            #     yatstop = y[-1][:,-1]
            # except:
            #     yatstop = y0

                

            # for model in models:
            #     model.compute_internal_flux(yatstop)
            #     print("Current {} internal flux norm: {} \n and max internal flux: {}".format(model.Name,np.linalg.norm(model.inter_flux),np.max(abs(model.inter_flux))))
            #     neg_flux = np.where(model.inter_flux < -10**-5)[0]
            #     for ng in neg_flux:
            #         print("Negative flux at index {}, value {}".format(ng,model.inter_flux[ng]))
            #     model.compute_slacks(yatstop)
            #     neg_slack = np.where(model.slack_vals < -10**-5)[0]
            #     for ng in neg_slack:
            #         print("Negative slack at constraint index {}, value {}".format(ng,model.slack_vals[ng]))
            

            # minmetind = np.argmin(s0[len(models):])
            # minmetval = s0[len(models):][minmetind]
            # print("Current Min met: index {}, value {}".format(minmetind,minmetval))
            # interval2 = solve_ivp(evolve_community, (t_c,endtime), s0, args=(models,), method='RK45',dense_output = True)
            # T2 = np.linspace(t_c,endtime,max(2,int((endtime-t_c)*(1/resolution))))
            # s2 = interval2.sol(T2)
            # y2 = s2[len(models):,:]
            # # x += [s2[:len(models)]]
            # # y += [y2]
            # # t += [T2]
            # negatives = [np.any(col < -10**-4) for col in y2.T]
            # if np.any(negatives):
            #     negcol = np.where(negatives)[0][0]
            #     negt = T2[negcol]
            #     negmetind = np.argmin(y2.T[negcol])
            #     negval = y2.T[negcol][negmetind]
            #     print("Negative Metabolites? index: {}, value: {}, time: {}".format(negmetind,negval,negt))
            # else:
            #     print("Negative Metabolites? No.")


            # print("No progress at time ",t_c)
            # for model in models:
            #     if solver == 'gurobi':
            #         obval = model.fba_gb(yatstop.round(5))
            #     elif solver == 'clp':
            #         obval = model.fba_clp(yatstop.round(5))
            #     model.compute_internal_flux(yatstop.round(5))
            #     print("Current {} growth rate {}".format(model.Name,obval))
            #     print("Current {} internal flux norm after re-FBA: {} \n and max internal flux: {}".format(model.Name,np.linalg.norm(model.inter_flux),np.max(abs(model.inter_flux))))

            #     ydi = np.zeros_like(ydot0)
            #     ydi[model.ExchangeOrder] = np.dot(model.GammaStar,model.inter_flux)
            #     print("Contribution of {} to ydot after re-FBA: {}".format(model.Name,np.linalg.norm(ydi)))

            else:
                print("No progress at time {}. Stopping Simulation.")
                break
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################

        s0 = interval.sol(stptime)
        T = np.linspace(t_c,stptime,max(2,int((stptime-t_c)*(1/resolution))))
        s = interval.sol(T)
        x += [s[:len(models),:]]
        y += [s[len(models):,:]]
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
                    flobj.write("surfin_fba: Finding New Basis at time ",t_c," \n")
                except:
                    print("surfin_fba: Finding New Basis at time ",t_c)
            yd = evolve_community(t_c,s0,models)[len(models):]
            basis_change += [t_c]
            for model in models:


                metabolite_con = y[-1][:,-1][model.ExchangeOrder]
                
                exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])

                bound_rhsarr = np.concatenate([exchg_bds,model.internal_bounds])

                model.compute_internal_flux(y[-1][:,-1])
                slkvals = bound_rhsarr - np.dot(model.solver_constraint_matrix,model.inter_flux)

                all_vars = np.concatenate([model.inter_flux,slkvals])

                model.essential_basis = (all_vars>model.ezero).nonzero()[0]


                model.findWave(y[-1][:,-1],yd,details = fwreport)
                if save_bases:
                    bases[model.Name] += [model.current_basis]

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
