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

import matplotlib.pyplot as plt


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


  models,metlist,y0dict = pr.prep_cobrapy_models(cobra_models,ub_funs = "linearScale",forceOns=True)#"linearRand")#hill11

  # y0dict['1,5-Diaminopentane'] =  0.9042818633990344
  # y0dict['Acetate'] =  0.6377115946128331
  # y0dict['Ammonium'] =  0.9680868002469349 
  # y0dict['CO2 CO2'] =  21.147031438774665 
  # y0dict['D-Glucose'] =  0.9527715413263937 
  # y0dict['H+'] =  6.725807277506377
  # y0dict['H2O H2O'] =  35.93636984219523 
  # y0dict['L-Alanine'] =  0.12408520385919787 
  # y0dict['O2 O2'] =  1.8987200578496777
  # y0dict['Phosphate'] =  0.24750125668547476 
  # y0dict['Sulfate'] =  0.06327342503867513

  y0dict['1,5-Diaminopentane'] =  0.5599251910752577  #initial: 0
  y0dict['Acetate'] =  2.6399966266628385 #0
  y0dict['Ammonium'] =  1.891684696081923 #9.850520309487253
  y0dict['CO2 CO2'] =  20.81322091083119 #0
  y0dict['D-Glucose'] =  0.7618404833201142  #10
  y0dict['Ethanol'] = 1.959576384069774*10**-6 #0
  y0dict['Formate'] = 2.095931684869389*10**-6 #0
  y0dict['H+'] =  8.491244066404017 #0
  y0dict['H2O H2O'] =  34.92232236698047 #0
  y0dict['L-Alanine'] =  0.0013694067412982423 #0 
  y0dict['O2 O2'] =  1.5182252447927205 #19.92838760912608
  y0dict['Phosphate'] =  0.2570719001940351 #0.8404819955275409
  y0dict['Sulfate'] =  0.06572014956331541 #0.21486830108436766


  print([(ky,val) for ky,val in y0dict.items() if abs(val)>0])
  # print(cobra_models["E.coli"].medium)
  # print(cobra_models["E.coli"].slim_optimize())

  y0 = np.array([y0dict[met] for met in metlist])
  ydot0 = np.zeros_like(y0)
  # fluxes = {}

  for ky,model in models.items():


    model.ezero  = 10**-8

  testInitial = False
  if testInitial:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Initial FBA \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for ky,model in models.items():


      model.ezero  = 10**-8

      print("Testing Initial: {}".format(ky))

      metabolite_con = y0[model.ExchangeOrder]
      exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
      bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

      obval = model.fba_gb(y0)#,secondobj = None)

      # obval = model.fba_clp(y0)#,secondobj = None)

      # print(ky," COBRA growth rate", cobra_models[ky].slim_optimize())
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
      print("Distance from basis flux to gurobi flux = {}".format(dist))

      ydi = np.zeros_like(ydot0)
      ydi[model.ExchangeOrder] = -np.dot(model.GammaStar,model.inter_flux)
      ydot0 += ydi

      # all_slv = np.concatenate([model.inter_flux, model.slack_vals])
      # neg_ind = np.where(all_slv<-10**-5)
      # if len(neg_ind[0]):
      #   for ind in neg_ind[0]:
      #     print("Negative flux from Internal at index {} = {}".format(ind,all_slv[ind]))      
      # else:
      #   print("No negative flux from Internal at Basis")


      # neg_ind = np.where(basisflxes<-10**-5)  
      # if len(neg_ind[0]):
      #   for ind in neg_ind[0]:
      #     print("Negative flux from Computed at index {} = {}".format(ind,basisflxes[ind]))
      # else:
      #   print("No negative flux from Computed")


  testWaves = False
  if testWaves:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Find Waves \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    for ky,model in models.items():
      
      print("Testing Findwaves: {}".format(ky))

      model.findWave(y0,ydot0,details = True)

      metabolite_con = y0[model.ExchangeOrder]
      exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
      bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

      incess = np.all([j in model.current_basis_full for j in model.essential_basis])
      print("Includes the essentials? {}".format(incess))
      ###
      rk = np.linalg.matrix_rank(model.standard_form_constraint_matrix[:,model.current_basis_full])
      print("Is full rank? {}".format(rk == model.standard_form_constraint_matrix.shape[0]))
      svd = np.linalg.svd(model.standard_form_constraint_matrix[:,model.current_basis_full],compute_uv = False)
      print("Minimum Singular Value: {}".format(min(svd)))

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
      print("Reduction error {}".format(np.linalg.norm(model.inter_flux-fluxesfound)))

      neg_ind = np.where(basisflxes<-10**-5)
      if len(neg_ind[0]):
        for ind in neg_ind[0]:
          print("Negative flux from Waves-Basis at index {} = {}".format(ind,basisflxes[ind]))
      else:
        print("No negative flux from Waves-Basis.")

  testNetwork = False
  if testNetwork:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Make Network \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    try:
      os.mkdir("test")
    except:
      pass
    rmnodes,rmfull,rmsumm,mmedges,mmnodes = mn.species_metabolite_network(metlist,y0,models)
    ssnet,ssnodes,ssadj=mn.heuristic_ss(rmsumm,rmnodes)
    rmnodes.to_csv("test/rmnodes.csv")
    rmfull.to_csv("test/rmfulledges.csv")
    rmsumm.to_csv("test/rmsummedges.csv")
    mmedges.to_csv("test/mmedges.csv")
    mmnodes.to_csv("test/mmndoes.csv")
    ssnet.to_csv("test/ssedges.csv")
    ssnodes.to_csv("test/ssnodes.csv")

  testSimulation = True
  if testSimulation:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Dynamic Simulation \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    dynami = surf.surfin_fba(list(models.values()),[1],y0,10,fwreport= True,solver = 'gurobi')#,solver = 'clp')
    # print(dynami['t'], '\n\n',dynami['x'])
    print("Final x: {}".format(dynami['x'][:,-1]))
    ydict = dict([(metlist[i],dynami['y'][i,-1]) for i in range(len(metlist))])
    print([(ky,val) for ky,val in ydict.items() if abs(val)>10**-6])
    plt.plot(dynami['t'],dynami['x'].T)
    for bt in dynami['bt']:
      plt.plot([bt]*10,np.linspace(np.min(dynami['x']),np.max(dynami['x']),10),":")
    plt.show()
    plt.plot(dynami['t'],dynami['y'].T)
    for bt in dynami['bt']:
      plt.plot([bt]*10,np.linspace(np.min(dynami['y']),np.max(dynami['y']),10),":")
    plt.show()


    # dyn2 = surf.surfin_fba(list(models.values()),[dynami['x'][0][-1]],dynami['y'][:,-1],10,fwreport=True)
    # print(dyn2['t'], '\n\n',dyn2['x'])

    minuts,sec = divmod(time.time() - t1, 60)

    print("Total Script Time: ",int(minuts)," minutes, ",sec," seconds.")


