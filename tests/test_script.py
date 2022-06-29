import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metconsin"))
from metconsin import *
import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    species = ["E.coli"]#["S.cerevisiae","M.tuberculosis","E.coli","P.putida"]#["E.coli"]#

    metconsin_return = metconsin(species,"bigg_model_file_info.txt",solver = 'clp',endtime = 10,track_fluxes = True,save_internal_flux = True,ub_funs = "linearScale",linearScale = 1.0,resolution = 0.01)#,media = "minimal")

    print(metconsin_return["BasisChanges"])

    tmlabel = dt.datetime.now()

    flder = os.path.join(os.path.expanduser("~"),"Documents","metconsin_networks","_".join([m.replace(".","") for m in species])+tmlabel.strftime("%a%d%H%M"))

    Path(flder).mkdir(parents=True, exist_ok=True)

    metconsin_return["Microbes"].to_csv(os.path.join(flder,"Microbes.tsv"),sep="\t")
    metconsin_return["Metabolites"].to_csv(os.path.join(flder,"Metabolites.tsv"),sep="\t")
    pd.DataFrame(metconsin_return["BasisChanges"]).to_csv(os.path.join(flder,"BasisTimes.tsv"),sep = "\t")
    for ky in metconsin_return["MetMetNetworks"].keys():
        metconsin_return["MetMetNetworks"][ky]["edges"].to_csv(os.path.join(flder,"MetMetEdges"+ky+".tsv"),sep="\t",index = False)
        metconsin_return["MetMetNetworks"][ky]["nodes"].to_csv(os.path.join(flder,"MetMetNodes"+ky+".tsv"),sep="\t")
        metconsin_return["SpcMetNetworkSummaries"][ky]["edges"].to_csv(os.path.join(flder,"SpcMetNetworksEdgesSummary"+ky+".tsv"),sep="\t",index = False)
        metconsin_return["SpcMetNetworks"][ky]["edges"].to_csv(os.path.join(flder,"SpcMetNetworksEdges"+ky+".tsv"),sep="\t",index = False)
        metconsin_return["SpcMetNetworks"][ky]["nodes"].to_csv(os.path.join(flder,"SpcMetNetworksNodes"+ky+".tsv"),sep="\t")

    for model in species:
        metconsin_return["ExchangeFluxes"][model].to_csv(os.path.join(flder,model.replace(".","")+"exchange_flux.tsv"),sep="\t")
        metconsin_return["InternalFluxes"][model].to_csv(os.path.join(flder,model.replace(".","")+"internal_flux.tsv"),sep="\t")

    x_pl = metconsin_return["Microbes"].copy()
    x_pl.columns = np.array(metconsin_return["Microbes"].columns).astype(float).round(4)
    ax = x_pl.T.plot(figsize = (20,10));
    for bt in metconsin_return["BasisChanges"]:
        ax.axvline(x = bt,linestyle = ":")
    ax.legend(prop={'size': 30},loc = 2)
    plt.savefig(os.path.join(flder,"Microbes.png"))

    y_pl = metconsin_return["Metabolites"].copy()
    y_pl.columns = np.array(metconsin_return["Metabolites"].columns).astype(float).round(4)
    ax = y_pl.loc[np.array([max(metconsin_return["Metabolites"].loc[nd]) >10**-6 for nd in metconsin_return["Metabolites"].index])].T.plot(figsize = (20,10));
    for bt in metconsin_return["BasisChanges"]:
        ax.axvline(x = bt,linestyle = ":")
    ax.legend(prop={'size': 15})
    plt.savefig(os.path.join(flder,"Metabolites.png"))

    for model in species:
        exchpl = metconsin_return["ExchangeFluxes"][model].copy()
        exchpl.columns = np.array(metconsin_return["ExchangeFluxes"][model].columns).astype(float).round(4)
        ax = exchpl.loc[np.array([max(metconsin_return["Metabolites"].loc[nd]) >10**-6 for nd in metconsin_return["Metabolites"].index])].T.plot(figsize = (20,10));
        for bt in metconsin_return["BasisChanges"]:
            ax.axvline(x = bt,linestyle = ":")
        ax.legend(prop={'size': 15})
        plt.savefig(os.path.join(flder,model + "Exchange.png"))
