import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","metconsin"))
import pandas as pd
from pathlib import Path
import datetime as dt
model_info = pd.read_csv("bigg_model_file_info.txt")
import pickle
import importlib as il
import cobra as cb
import numpy as np
import make_network as mn
import time


import surfmod as sm
import prep_models as pr
import dynamic_simulation as surf

if __name__ == "__main__":

  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Set Up \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

  t1 = time.time()

  cbmods = {}

  desired_models = ["E.coli"]#,"P.putida"]#,"S.cerevisiae","M.tuberculosis",

  cobra_models = {}

  for mod in desired_models:
    if any(model_info.Species == mod):
      flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
      cobra_models[mod] = cb.io.load_json_model(flnm)
      if not cobra_models[mod].name:
        cobra_models[mod].name = mod
      else:
        print("Error: No model of species " + mod)
      mxg = cobra_models[mod].slim_optimize()
      min_med = cb.medium.minimal_medium(cobra_models[mod],mxg)#,minimize_components = True)
      cobra_models[mod].medium = min_med


  models,metlist,y0dict = pr.prep_cobrapy_models(cobra_models,ub_funs = "linearScale")#"linearRand")#
  print([(ky,val) for ky,val in y0dict.items() if val>0])
  # print(cobra_models["E.coli"].medium)
  # print(cobra_models["E.coli"].slim_optimize())

  y0 = np.array([y0dict[met] for met in metlist])
  ydot0 = np.zeros_like(y0)
  # fluxes = {}
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Initial FBA \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

  for ky,model in models.items():
    metabolite_con = y0[model.ExchangeOrder]
    exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
    bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

    obval = model.fba_clp(y0)#,secondobj = None)


    print(ky," COBRA growth rate", cobra_models[ky].slim_optimize())
    print(ky," growth rate: ",obval)

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
    print("Distance from basis flux to CyLP flux = {}".format(dist))

    ydi = np.zeros_like(ydot0)
    ydi[model.ExchangeOrder] = -np.dot(model.GammaStar,model.inter_flux)
    ydot0 += ydi


  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Find Waves \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

  for ky in models:
    
    models[ky].findWave(y0,ydot0)


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
    print("Distance from basis flux to CyLP flux = {}".format(dist))

    ###Let's check the reduced as well.
    model.compute_internal_flux(y0)
    print("Reduction error {}".format(np.linalg.norm(model.inter_flux-fluxesfound)))

      # basesFluxes = models[ky].compute_internal_flux(y0)

      # print(ky," basis error: ",np.linalg.norm(fluxes[ky]-basesFluxes))
  
  # #
  # model_list = [model for model in models.values()]
  # x0 = np.array([1 for i in range(len(model_list))])
  # dynamics = surf.surfin_fba(model_list,x0,y0,0)
  # dynamics["Metabolites"] = metlist
  # dynamics["Models"] = [model.Name for model in model_list]
  #
  #
  # for model in models.keys():
  #     tmpmed = {}
  #     exchng_reactions = [rxn.id for rxn in cobra_models[model].reactions if "EX_" in rxn.id]
  #     reactants = [[m.name for m in cobra_models[model].reactions.get_by_id(rxn).reactants] for rxn in exchng_reactions]
  #     for m in range(len(dynamics["Metabolites"])):#for rxn in tmpmed.keys():
  #         met = dynamics["Metabolites"][m]
  #         #find the reaction
  #         rindex = np.where([met in rs for rs in reactants])
  #         rid = np.array(exchng_reactions)[rindex][0]
  #         tmpmed[rid] = dynamics["y"][m,-1]
  #         # rcts = [r.name for r in cobra_models[model].reactions.get_by_id(rxn).reactants][0]
  #         # tmpmed[rxn] = dynamics["y"][np.where(dynamics["Metabolites"]==rcts),-1][0][0]
  #     cobra_models[model].medium = tmpmed
  #     # print(cobra_models[model].medium)
  #     print("Cobra FBA at end: ",cobra_models[model].slim_optimize())
  #     # v,opt = models[model].fba_clp(dynamics["y"][:,-1])
  #     # print(opt)
  #
  #
  # fluxes = {}
  # ydot0 = np.zeros_like(dynamics["y"][:,-1])
  # for ky,model in models.items():
  #     flux,obval = model.fba_clp(dynamics["y"][:,-1])#,secondobj = None)
  #     # print(ky," COBRA growth rate", cobra_models[ky].slim_optimize())
  #     print(ky," growth rate: ",obval)
  #     fluxes[ky] = flux
  #     ydi = np.zeros_like(ydot0)
  #     ydi[model.ExchangeOrder] = -np.dot(np.concatenate([model.GammaStar,-model.GammaStar],axis = 1),flux[:2*model.num_fluxes])
  #     ydot0 += ydi
  #
  # for ky in models:
  #     try:
  #         models[ky].find_waves_gb(fluxes[ky],dynamics["y"][:,-1],ydot0)
  #
  #         basesFluxes = models[ky].compute_internal_flux(dynamics["y"][:,-1])
  #
  #         print(ky," basis error: ",np.linalg.norm(fluxes[ky]-basesFluxes))
  #
  #     except:
  #         pass
  #
  #
  #
  #
  #
  # #NEXT: GET THE NETWORK
  # # node_table,met_med_net,met_med_net_summary,met_met_edges,met_met_nodes = mn.species_metabolite_network(metlist,y0,models,report_activity = False)
  # tmlabel = dt.datetime.now()
  #
  # flder = os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","_".join([m.replace(".","") for m in desired_models])+tmlabel.strftime("%a%d%H%M"))
  #
  # Path(flder).mkdir(parents=True, exist_ok=True)
  # #
  # #
  # #
  # # node_table.to_csv(os.path.join(flder,"smi_nodes.tsv"),sep="\t")
  # # met_med_net.to_csv(os.path.join(flder,"cof_edges.tsv"),sep="\t")
  # # met_med_net_summary.to_csv(os.path.join(flder,"sum_edges.tsv"),sep="\t")
  # # met_met_edges.to_csv(os.path.join(flder,"met_met_edges.tsv"),sep="\t")
  # # met_met_nodes.to_csv(os.path.join(flder,"met_met_nodes.tsv"),sep="\t")
  # #THEN: DYNAMIC SIMULATION
  #
  #
  #
  # with open(os.path.join(flder,'dynamics.pickle'), 'wb') as handle:
  #     pickle.dump(dynamics, handle, protocol=pickle.HIGHEST_PROTOCOL)

  #FINALLY HELPER/WRAPPER FUNCTIONS



  minuts,sec = divmod(time.time() - t1, 60)

  print("Total Script Time: ",int(minuts)," minutes, ",sec," seconds.")
