import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","metconsin"))
from metconsin import *
import pandas as pd
from pathlib import Path
import datetime as dt

if __name__=="__main__":
    species = ["E.coli"]#["S.cerevisiae","M.tuberculosis","E.coli","P.putida"]

    metconsin_return = metconsin(species,"bigg_model_file_info.txt",endtime = 10,track_fluxes = True,save_internal_flux = True,ub_funs = "linear1")

    print(metconsin_return["BasisChanges"])

    tmlabel = dt.datetime.now()

    flder = os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","_".join([m.replace(".","") for m in species])+tmlabel.strftime("%a%d%H%M"))

    Path(flder).mkdir(parents=True, exist_ok=True)

    metconsin_return["Microbes"].to_csv(os.path.join(flder,"Microbes.csv"))
    metconsin_return["Metabolites"].to_csv(os.path.join(flder,"Metabolites.csv"))
    for ky in metconsin_return["MetMetNetworks"].keys():
        metconsin_return["MetMetNetworks"][ky]["edges"].to_csv(os.path.join(flder,"MetMetEdges"+ky+".csv"))
        metconsin_return["MetMetNetworks"][ky]["nodes"].to_csv(os.path.join(flder,"MetMetNodes"+ky+".csv"))
        metconsin_return["SpcMetNetworkSummaries"][ky]["edges"].to_csv(os.path.join(flder,"SpcMetNetworksEdgesSummary"+ky+".csv"))
        metconsin_return["SpcMetNetworks"][ky]["edges"].to_csv(os.path.join(flder,"SpcMetNetworksEdges"+ky+".csv"))
        metconsin_return["SpcMetNetworks"][ky]["nodes"].to_csv(os.path.join(flder,"SpcMetNetworksNodes"+ky+".csv"))

    for model in species:
        metconsin_return["ExchangeFluxes"][model].to_csv(os.path.join(flder,model.replace(".","")+"exchange_flux.csv"))
        metconsin_return["InternalFluxes"][model].to_csv(os.path.join(flder,model.replace(".","")+"internal_flux.csv"))
