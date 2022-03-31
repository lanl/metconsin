import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","metconsin"))
import pandas as pd
from pathlib import Path
import datetime as dt
model_info = pd.read_csv("bigg_model_file_info.txt")
import surfmod as sm
import prep_models as pr
import importlib as il
import cobra as cb
import numpy as np
import make_network as mn
import time


if __name__ == "__main__":

    t1 = time.time()

    cbmods = {}

    desired_models = ["S.cerevisiae"]#["M.tuberculosis","E.coli","P.putida"]#,

    cobra_models = {}

    for mod in desired_models:
      if any(model_info.Species == mod):
        flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
        cobra_models[mod] = cb.io.load_json_model(flnm)
        if not cobra_models[mod].name:
          cobra_models[mod].name = mod
        else:
          print("Error: No model of species " + mod)


    models,metlist,y0dict = pr.prep_cobrapy_models(cobra_models,ub_funs = "linear1")#"linearRand")#


    y0 = np.array([y0dict[met] for met in metlist])
    ydot0 = np.zeros_like(y0)
    fluxes = {}
    for ky,model in models.items():
        flux,obval = model.fba_gb(y0)#,secondobj = None)
        print(obval)
        fluxes[ky] = flux
        ydot0 += -np.dot(np.concatenate([model.GammaStar,-model.GammaStar],axis = 1),flux[:2*model.num_fluxes])

    for ky in models:
        models[ky].find_waves_gb(fluxes[ky],y0,ydot0)

        basesFluxes = models[ky].compute_flux(y0)

        print(ky," basis error: ",np.linalg.norm(fluxes[ky]-basesFluxes))


    #NEXT: GET THE NETWORK
    # node_table,met_med_net,met_med_net_summary = mn.species_metabolite_network(metlist,y0,models,report_activity = False)
    # x = dt.datetime.now()
    #
    # flder = os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","testNetwork"+x.strftime("%a%d%H%M"))
    #
    # Path(flder).mkdir(parents=True, exist_ok=True)
    #
    #
    #
    # node_table.to_csv(os.path.join(flder,"smi_nodes.tsv"),sep="\t")
    # met_med_net.to_csv(os.path.join(flder,"cof_edges.tsv"),sep="\t")
    # met_med_net_summary.to_csv(os.path.join(flder,"sum_edges.tsv"),sep="\t")
    #THEN: DYNAMIC SIMULATION

    #FINALLY HELPER/WRAPPER FUNCTIONS



    minuts,sec = divmod(time.time() - t1, 60)

    print("Total Script Time: ",int(minuts)," minutes, ",sec," seconds.")
