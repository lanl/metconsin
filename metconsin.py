import make_network as mn
import surfmod as sm
import prep_models as pr
import dynamic_simulation as surf
import time
import pandas as pd
import cobra as cb
import numpy as np
import contextlib
import os


def metconsin_network(desired_models,
    model_info_file, 
    save_folder,
    media = None,
    metabolite_inflow = None,
    metabolite_outflow = None,
    solver = 'gurobi',
    report_activity = True,
    upper_bound_functions = None,
    lower_bound_functions = None,
    upper_bound_functions_dt = None,
    lower_bound_functions_dt = None,
    met_filter = None,
    met_filter_sense = "exclude", 
    lb_funs = "constant", 
    ub_funs = "linearRand",
    linearScale=1.0,
    secondobj = "total",
    flobj = None,
    trimit = True,
    forceOns = True,
    report_activity_network = True,
    returnNets = False
    ):
    '''


    Creates dFBA implied network (microbe-metabolite, metabolite-metabolite, heuristic microbe-microbe) for community interaction from GEMs with media that reflect the desired environment.
    If no media is provided, the default ``diet" files for the given COBRA models is averaged.

    '''

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


    start_time = time.time()

    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in desired_models:
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
        media_input = {}
    else:
        media_input = media

    for mod in cobra_models.keys():
        print(mod," COBRA initial growth rate: ",cobra_models[mod].slim_optimize())

    #returns dict of surfmods, list of metabolites, and concentration of metabolites.
    # models,mets,mets0 = prep_cobrapy_models(cobra_models,uptake_dicts = uptake_dicts ,random_kappas=random_kappas)




    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,
                                                    upper_bound_functions = upper_bound_functions,
                                                    lower_bound_functions = lower_bound_functions,
                                                    upper_bound_functions_dt = upper_bound_functions_dt,
                                                    lower_bound_functions_dt = lower_bound_functions_dt,
                                                    media = media_input, 
                                                    met_filter = met_filter,
                                                    met_filter_sense = met_filter_sense, 
                                                    lb_funs = lb_funs, 
                                                    ub_funs = ub_funs,
                                                    linearScale=linearScale,
                                                    forceOns=forceOns)

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

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Compute FBA \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Find Waves \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    for ky,model in models.items():
        
        print("Findwaves: {}".format(ky))

        model.findWave(y0,ydot0,details = report_activity,flobj = flobj)

        model.compute_internal_flux(y0)


    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Make Network \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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



def metconsin_sim(desired_models,
    model_info_file,
    final_interval_weight = 0.1,
    metabolite_inflow = None,
    metabolite_outflow = None,
    model_deathrates = None,
    solver = 'gurobi',
    flobj = None,
    endtime = 10**-2,
    upper_bound_functions = None,
    lower_bound_functions = None,
    upper_bound_functions_dt = None,
    lower_bound_functions_dt = None,
    media = None, 
    met_filter = None,
    met_filter_sense = "exclude", 
    lb_funs = "constant", 
    ub_funs = "linearRand",
    linearScale=1.0,
    track_fluxes = True,
    save_internal_flux = True, 
    resolution = 0.1,
    report_activity = True,
    report_activity_network=True,
    forceOns = True,
    findwaves_report = False,
    debugging = False):
    '''

    Does Full Simulation, makes networks for each time interval.

    '''
    start_time = time.time()


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


    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in desired_models:
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
        media_input = {}
    else:
        media_input = media

    for mod in cobra_models.keys():
        cbgr = cobra_models[mod].slim_optimize()
        try:
            flobj.write("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))
        except:
            print("[MetConSIN] {} COBRA initial growth rate: {}\n".format(mod,cbgr))

    #returns dict of surfmods, list of metabolites, and concentration of metabolites.
    # models,mets,mets0 = prep_cobrapy_models(cobra_models,uptake_dicts = uptake_dicts ,random_kappas=random_kappas)

    models,metlist,y0dict =  pr.prep_cobrapy_models(cobra_models,
                                                    deathrates=model_deathrates,
                                                    upper_bound_functions = upper_bound_functions,
                                                    lower_bound_functions = lower_bound_functions,
                                                    upper_bound_functions_dt = upper_bound_functions_dt,
                                                    lower_bound_functions_dt = lower_bound_functions_dt,
                                                    media = media_input, 
                                                    met_filter = met_filter,
                                                    met_filter_sense = met_filter_sense, 
                                                    lb_funs = lb_funs, 
                                                    ub_funs = ub_funs,
                                                    linearScale=linearScale,
                                                    forceOns=forceOns,
                                                    flobj=flobj)

    #for new network after perturbing metabolites, we only need to update mets0.
    #mets establishes an ordering of metabolites.
    #Next establish an ordering of microbes. Note, this won't necessarily be predictable, python
    #dict keys are unordered and calling dict.keys() will give whatever ordering it wants.

    y0 = np.array([y0dict[met] for met in metlist])

    model_list = [model for model in models.values()]
    x0 = np.array([1 for i in range(len(model_list))])



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

    all_sim = pd.concat([x_sim,y_sim])

    met_met_nets = {}
    mic_met_sum_nets = {}
    mic_met_nets = {}
    speciesHeuristic = {}
    interval_lens = {}
    total_interval = 0

    for i in range(len(basis_change_times)):
        #get the times - if we're on the last one there's no "end time"
        t0 = basis_change_times[i]
        try:
            t1 = basis_change_times[i+1]
            ky = "{:.4f}".format(t0)+"-"+"{:.4f}".format(t1)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))&(np.array(all_sim.columns.astype(float)).round(6)<=t1)]
            interval_lens[ky] = t1-t0
        except:
            ky = "{:.4f}".format(t0)
            dynamics_t = all_sim.loc[:,(t0 <= np.array(all_sim.columns.astype(float)).round(6))]
            interval_lens[ky] = final_interval_weight
            total_interval = t0/(1-final_interval_weight)


        if dynamics_t.shape[1]:#skip if the interval is too small to contain any of the dynamics (this has to be due to roundoff error - there should be a point at each basis change time.)
            for model in model_list:
                #dynamics["basis"][model.Name] is a list of tuples of (basis change time, index of reduced basis - rows/columns)
                modbc = [bc[0] for bc in dynamics["bases"][model.Name]]#list of times this model changed basis
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
                ssnet,ssnodes,ssadj=mn.heuristic_ss(met_med_net_summary,node_table,report_activity=report_activity_network)

                met_met_nets[ky] = {"nodes":met_met_nodes,"edges":met_met_edges}
                mic_met_sum_nets[ky] = {"nodes":node_table,"edges":met_med_net_summary}
                mic_met_nets[ky] = {"nodes":node_table,"edges":met_med_net}
                speciesHeuristic[ky] = {"nodes":ssnodes,"edges":ssnet,"adjacency":ssadj}
    
    avg_micmetnet_sum,avg_micmet_summ_nodes = mn.average_network(mic_met_sum_nets,interval_lens,total_interval,"micmet")
    if avg_micmetnet_sum != None:
        mic_met_sum_nets["Combined"] = {"nodes":avg_micmet_summ_nodes,"average_edges":avg_micmetnet_sum}

    avg_micmetnet,avg_micmet_nodes = mn.average_network(mic_met_nets,interval_lens,total_interval,"micmet")
    if avg_micmetnet != None:
        mic_met_nets["Combined"] = {"nodes":avg_micmet_nodes,"average_edges":avg_micmetnet}

    avg_metmetnet,avg_metmet_nodes = mn.average_network(met_met_nets,interval_lens,total_interval,"metmet")
    if avg_metmetnet != None:
        met_met_nets["Combined"] = {"nodes":avg_metmet_nodes,"average_edges":avg_metmetnet}

    avg_spec,avg_spc_nodes = mn.average_network(speciesHeuristic,interval_lens,total_interval,"spc")
    if avg_spec != None:
        speciesHeuristic["Combined"] = {"nodes":avg_spc_nodes,"average_edges":avg_spec}



    all_return = {"Microbes":x_sim,"Metabolites":y_sim,"SpeciesNetwork":speciesHeuristic,"MetMetNetworks":met_met_nets, "SpcMetNetworkSummaries":mic_met_sum_nets,"SpcMetNetworks":mic_met_nets, "BasisChanges":basis_change_times}

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

