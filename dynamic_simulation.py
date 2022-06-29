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
        v = model.compute_internal_flux(y)
        xdot[i] = x[i]*np.dot(-model.objective,v) - model.deathrate
        ydi = np.zeros_like(ydot)
        ydi[model.ExchangeOrder] = -x[i]*np.dot(model.expandGammaStar,v)#Influx = GammaStar*v
        ydot = ydot + ydi

    return np.concatenate([xdot,ydot])

def all_vs(t,s,models):
    y = s[len(models):]
    v = np.array([])
    for i in range(len(models)):
        model = models[i]
        v = np.append(v,model.compute_internal_flux(y))#[model.current_basis[2]]
    return min(v) + 10**-6

def find_stop(t0,t1,sln,models,cdt = 0.1,fdt=0.001):
    #check on a coarse grid for negatives:
    tcrs = np.arange(t0,t1,cdt)
    minpert = np.array([all_vs(t,sln(t),models) for t in tcrs])
    if minpert.round(4).min() >= 0:
        return t1
    else:
        #find first negative
        failat = np.where(minpert < 0)[0][0]
        #go finer.
        if failat == 0:
            return t0
        else:
            finerfail = np.array([all_vs(t,sln(t),models) for t in np.arange(tcrs[failat-1],tcrs[failat],fdt)])
            return np.where(minpert < 0)[0][0]

def surfin_fba(models,x0,y0,endtime, solver = 'gurobi',track_fluxes = True, save_bases = True,save_internal_flux = True, resolution = 0.1,report_activity = True, flobj = None):
    t1 = time.time()
    if report_activity:
        try:
            flobj.write("surfin_fba: Initializing Simulation\n")
        except:
            print("surfin_fba: Initializing Simulation")

    model_names = [model.Name for model in models]
    s0 = np.concatenate([x0,y0])
    ydot0 = np.zeros_like(y0)
    fluxes = {}
    for i in range(len(models)):#model in models:
        model = models[i]
        if solver == 'gurobi':
            flux,obval = model.fba_gb(y0)#,secondobj = None)
        elif solver == 'clp':
            flux,obval = model.fba_clp(y0)
        if report_activity:
            try:
                flobj.write(model.Name, " Surfin initial growth rate: ",obval)
            except:
                print(model.Name, " Surfin initial growth rate: ",obval)
        fluxes[model.Name] = flux
        ydi = np.zeros_like(ydot0)
        ydi[model.ExchangeOrder] = -x0[i]*np.dot(model.expandGammaStar,flux)
        ydot0 += ydi

    for model in models:
        if solver == 'gurobi':
            model.find_waves_gb(fluxes[model.Name],y0,ydot0)
        elif solver == 'clp':
            model.find_waves_clp(fluxes[model.Name],y0,ydot0)

    if save_bases:
        bases = dict([(model.Name,[]) for model in models])
        for model in models:
            bases[model.Name] += [model.current_basis]

    if report_activity:
        bInitial = evolve_community(0,s0,models)
        try:
            flobj.write([model.Name for model in models], " Bases intial growth rate :", bInitial[:len(models)])
        except:
            print([model.Name for model in models], " Bases intial growth rate :", bInitial[:len(models)])

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

    while t_c < endtime:
        # infeasible_prob = lambda t,s,models : neg_v(t,s,models)
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
        interval = solve_ivp(evolve_community, (t_c,endtime), s0, args=(models,), method='RK45',events=all_vs,dense_output = True)
        if interval.status == -1:
            break
        stptime = find_stop(t_c, interval.t[-1],interval.sol,models)#interval.t[-1]#find_event(interval,models)
        if stptime == t_c:
            print("No progress at time ",t_c)
            break
        s0 = interval.sol(stptime)
        T = np.linspace(t_c,stptime,max(2,int((stptime-t_c)*(1/resolution))))
        s = interval.sol(T)
        x += [s[:len(models),:]]
        y += [s[len(models):,:]]
        if track_fluxes:
            for model in models:
                for i in range(y[-1].shape[1]):
                    flxt = np.zeros_like(y0)
                    flxt[model.ExchangeOrder] = np.dot(model.expandGammaStar,model.compute_internal_flux(y[-1][:,i]))
                    Exchfluxes[model.Name] += [flxt]
        if save_internal_flux:
            for model in models:
                intfluxes[model.Name] += [model.compute_internal_flux(y[-1][:,i]) for i in range(y[-1].shape[1])]
                # print(np.array(intfluxes[model.Name]).min())
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
                current_flux = model.compute_internal_flux(y[-1][:,-1])
                if solver == 'gurobi':
                    model.find_waves_gb(current_flux,y[-1][:,-1],yd,careful = True)
                elif solver == 'clp':
                    model.find_waves_clp(current_flux,y[-1][:,-1],yd,careful = True)
                if save_bases:
                    bases[model.Name] += [model.current_basis]



    retdict = {"t":np.concatenate(t),"x":np.concatenate(x,axis = 1),"y":np.concatenate(y,axis = 1),"bt":np.array(basis_change)}
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
