import make_network as mn
import surfmod as sm
import prep_models as pr
import dynamic_simulation as surf
import time
import pandas as pd
import cobra as cb
import numpy as np

def metconsin(desired_models,model_info_file,flobj = None,endtime = 10**-2,upper_bound_functions = {},lower_bound_functions = {},upper_bound_functions_dt = {},lower_bound_functions_dt = {},extracell = 'e', random_kappas = "new",media = {}, met_filter = [],met_filter_sense = "exclude", lb_funs = "constant", ub_funs = "linearRand",track_fluxes = True,save_internal_flux = True, resolution = 0.1,report_activity = True):
    '''

    Need to Doc....

    '''
    start_time = time.time()

    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in desired_models:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            cobra_models[mod] = cb.io.load_json_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)


    print("Loaded " + str(len(cobra_models)) + " models successfully")



    if media == "minimal":
        for model in cobra_models.keys():
            mxg = cobra_models[model].slim_optimize()
            min_med = cb.medium.minimal_medium(cobra_models[model],mxg,minimize_components = True)
            cobra_models[model].medium = min_med
    else:
        media_input = media


    #returns dict of surfmods, list of metabolites, and concentration of metabolites.
    # models,mets,mets0 = prep_cobrapy_models(cobra_models,uptake_dicts = uptake_dicts ,random_kappas=random_kappas)
    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,upper_bound_functions = upper_bound_functions,lower_bound_functions = lower_bound_functions,upper_bound_functions_dt = upper_bound_functions_dt,lower_bound_functions_dt = lower_bound_functions_dt,extracell = extracell, random_kappas = random_kappas,media = media_input, met_filter = met_filter,met_filter_sense = met_filter_sense, lb_funs = lb_funs, ub_funs = ub_funs,flobj = flobj)

    #for new network after perturbing metabolites, we only need to update mets0.
    #mets establishes an ordering of metabolites.
    #Next establish an ordering of microbes. Note, this won't necessarily be predictable, python
    #dict keys are unordered and calling dict.keys() will give whatever ordering it wants.

    y0 = np.array([y0dict[met] for met in metlist])

    model_list = [model for model in models.values()]
    x0 = np.array([1 for i in range(len(model_list))])
    dynamics = surf.surfin_fba(model_list,x0,y0,endtime,save_bases = True,track_fluxes = track_fluxes,save_internal_flux = save_internal_flux, resolution = resolution,report_activity = report_activity, flobj = flobj)
    # dynamics["Metabolites"] = metlist
    # dynamics["Models"] = [model.Name for model in model_list]

    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Simulation & Bases computed in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Simulation & Bases computed in ",int(minuts)," minutes, ",sec," seconds.")

    x_sim = pd.DataFrame(dynamics["x"].round(7),columns = dynamics["t"],index = [model.Name for model in model_list])
    x_sim = x_sim.loc[:,~x_sim.columns.duplicated()]
    y_sim = pd.DataFrame(dynamics["y"].round(7),columns = dynamics["t"],index = metlist)
    y_sim = y_sim.loc[:,~y_sim.columns.duplicated()]
    basis_change_times = dynamics["bt"]

    all_sim = pd.concat([x_sim,y_sim])

    met_met_nets = {}
    mic_met_sum_nets = {}
    mic_met_nets = {}
    for i in range(len(basis_change_times)):
        t0 = basis_change_times[i]
        try:
            t1 = basis_change_times[i+1]
            ky = "{:.4f}".format(t0)+"-"+"{:.4f}".format(t1)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))&(np.array(all_sim.columns.astype(float)).round(6)<=t1)]
        except:
            ky = "{:.4f}".format(t0)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))]
        for model in model_list:
            if dynamics["bases"][model.Name][i] != None:
                model.current_basis = dynamics["bases"][model.Name][i]
                bases_ok = True
            else:
                bases_ok = False
                break
        if bases_ok:
            metcons = y_sim.loc[:,t0].values
            node_table,met_med_net,met_med_net_summary,met_met_edges,met_met_nodes = mn.species_metabolite_network(metlist,metcons,model_list,report_activity = report_activity,flobj = flobj)

            #trim out species & metabolites that aren't present in this time interval.

            met_met_edges,met_met_nodes = trim_network(met_met_edges,met_met_nodes,dynamics_t)
            met_med_net_summary,node_table = trim_network(met_met_edges,met_met_nodes,dynamics_t)
            met_med_net_summary,node_table = trim_network(met_met_edges,met_met_nodes,dynamics_t)

            met_met_nets[ky] = {"nodes":met_met_nodes,"edges":met_met_edges}
            mic_met_sum_nets[ky] = {"nodes":node_table,"edges":met_med_net_summary}
            mic_met_nets[ky] = {"nodes":node_table,"edges":met_med_net}

    all_return = {"Microbes":x_sim,"Metabolites":y_sim, "MetMetNetworks":met_met_nets, "SpcMetNetworkSummaries":mic_met_sum_nets,"SpcMetNetworks":mic_met_nets, "BasisChanges":basis_change_times}

    if track_fluxes:
        exchg_fluxes = {}
        for model in model_list:
            eflux = pd.DataFrame(dynamics["Exchflux"][model.Name].round(7), columns = dynamics["t"],index = metlist)
            eflux = eflux.loc[:,~eflux.columns.duplicated()]
            exchg_fluxes[model.Name] = eflux
        all_return["ExchangeFluxes"] = exchg_fluxes

    if save_internal_flux:
        internal_flux = {}
        for model in model_list:
            iflux = pd.DataFrame(dynamics["Intflux"][model.Name].round(7)[:len(model.flux_order)], columns = dynamics["t"],index = model.flux_order)
            iflux = iflux.loc[:,~iflux.columns.duplicated()]
            internal_flux[model.Name] = iflux
        all_return["InternalFluxes"] = internal_flux


    minuts,sec = divmod(time.time() - start_time, 60)
    try:
        flobj.write("[MetConSIN]: Complete in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    except:
        print("[MetConSIN] Complete in ",int(minuts)," minutes, ",sec," seconds.")

    return all_return

def trim_network(edges,nodes,dynamics):
    newnodes = nodes.copy()
    newedges = edges.copy()
    for nd in dynamics.index:
        if nd in newnodes.index:
            if max(dynamics.loc[nd]) < 10**-6:
                newnodes.drop(index = nd, inplace = True)
                newedges = newedges.loc[(newedges["Source"] != nd) & (newedges["Target"] != nd)]
    return newedges,newnodes
