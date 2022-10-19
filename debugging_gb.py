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

  # y0dict['1,5-Diaminopentane']= 0.5599080693329191
  # y0dict['Acetate'] = 2.2498812211921333
  # y0dict['Ammonium'] = 1.8930883463079522
  # y0dict['CO2 CO2'] = 20.4230557410907
  # y0dict['D-Glucose'] = 0.9576064541037695
  # y0dict['H+'] = 8.099757158260273
  # y0dict['H2O H2O'] = 34.530772628332635
  # y0dict['O2 O2'] = 1.9083552594380746
  # y0dict['Phosphate'] = 0.25707190019403686
  # y0dict['Sulfate'] =  0.06572014956331518

  y0dict['1,5-Diaminopentane'] =  0.5599251910752636
  y0dict['Acetate'] = 2.639586621363106
  y0dict['Ammonium'] = 1.8916846334573647 
  y0dict['CO2 CO2'] = 20.812812517188462
  y0dict['D-Glucose'] = 0.7620456967867254
  y0dict['Ethanol'] = 1.4753016913259338e-06
  y0dict['H+'] = 8.490832027797154 
  y0dict['H2O H2O'] = 34.92191452022035 
  y0dict['L-Alanine'] = 0.001369469365858587
  y0dict['O2 O2'] = 1.5186342021432442
  y0dict['Phosphate'] = 0.2570719001940361 
  y0dict['Sulfate'] = 0.06572014956331566

  print([(ky,val) for ky,val in y0dict.items() if abs(val)>0])
  # print(cobra_models["E.coli"].medium)
  # print(cobra_models["E.coli"].slim_optimize())

  y0 = np.array([y0dict[met] for met in metlist])
  ydot0 = np.zeros_like(y0)
  # fluxes = {}

  testInitial = False
  if testInitial:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n Initial FBA \n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for ky,model in models.items():


      model.ezero =  10**-7

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

      all_slv = np.concatenate([model.inter_flux, model.slack_vals])
      neg_ind = np.where(all_slv<-10**-5)
      if len(neg_ind[0]):
        for ind in neg_ind[0]:
          print("Negative flux from Gurobi Basis at index {} = {}".format(ind,all_slv[ind]))      
      else:
        print("No negative flux from Gurobi Basis")


      neg_ind = np.where(basisflxes<-10**-5)  
      if len(neg_ind[0]):
        for ind in neg_ind[0]:
          print("Negative flux from Basis at index {} = {}".format(ind,basisflxes[ind]))
      else:
        print("No negative flux from Basis")


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

    dynami = surf.surfin_fba(list(models.values()),[1],y0,10,fwreport= True,solver = 'clp')#solver = 'gurobi'
    # print(dynami['t'], '\n\n',dynami['x'])

    ydict = dict([(metlist[i],dynami['y'][i,-1]) for i in range(len(metlist))])
    print([(ky,val) for ky,val in ydict.items() if abs(val)>10**-6])
    plt.plot(dynami['t'],dynami['x'].T)
    plt.show()
    plt.plot(dynami['t'],dynami['y'].T)
    plt.show()


    # dyn2 = surf.surfin_fba(list(models.values()),[dynami['x'][0][-1]],dynami['y'][:,-1],10,fwreport=True)
    # print(dyn2['t'], '\n\n',dyn2['x'])

    minuts,sec = divmod(time.time() - t1, 60)

    print("Total Script Time: ",int(minuts)," minutes, ",sec," seconds.")


