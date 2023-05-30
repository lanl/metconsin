import sys
import os
import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sb
import contextlib
import cobra as cb


from matplotlib.colors import ListedColormap

from itertools import combinations as com

import seaborn as sn

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from metconsin import metconsin_sim,save_metconsin,make_media
from metconsin import analysis_helpers as ah


if __name__=="__main__":

    model_info_fl = "ModelSeed_info.csv"

    et = 2.5

    metabolite_id_type = "modelSeedID"

    model_info_fl = "ModelSeed_info.csv"

    species = ['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']

    agora_flder = "AGORA_Media"
    agora_media_loc = os.path.join(parent,agora_flder)

    cobra_models = {}

    model_info = pd.read_csv(model_info_fl)

    for mod in species:
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

    if len(sys.argv) > 1:
        agora_media_nm = sys.argv[1]
    else:
        agora_media_nm = ""
        

    if "{}_AGORA.tsv".format(agora_media_nm) in os.listdir(agora_media_loc):
        agora_media_nm = "{}_AGORA.tsv".format(agora_media_nm)
        

    if agora_media_nm in os.listdir(agora_media_loc):
        agora_media = pd.read_csv(os.path.join(agora_media_loc,agora_media_nm),index_col = 0, sep = '\t')
        growth_media = make_media(cobra_models,media_df = agora_media,metabolite_id_type="modelSeedID").to_dict()
        mednm = agora_media_nm.split(".")[0]
    elif agora_media_nm == "minimal":
        print("Using minimal media.")
        growth_media = make_media(cobra_models,default_proportion = 1,minimal=True,minimal_grth=10).to_dict()
        mednm = "minimal"
    else:
        if agora_media_nm != "":
            print("Cannot find media file {}, using default environment from model mediums".format(os.path.join(agora_media_loc,agora_media_nm)))
        else:
            print("Using default model medias.")
        growth_media = make_media(cobra_models,default_proportion = 0.1).to_dict()
        mednm = "Default"


    with open("exchange_bounds_uniform.json") as fl:
        uptake_params = json.load(fl)

    tmlabel = dt.datetime.now()

    flder = "ExampleResults_{}_{}".format(mednm,tmlabel.strftime("%a%B%d_%Y_%H.%M"))

    # We can change the environment by metabolite ID
    # growth_media["D-Glucose_e0"] = 10
    # growth_media["O2_e0"] = 10


    Path(flder).mkdir(parents=True, exist_ok=True)

    oricmap = plt.cm.tab10.colors
    cdict = dict([(species[i],oricmap[i]) for i in range(len(species))])

    styles = ['o','v','^','>','<','s','P','*','X','D']
    sdict = dict([(species[i],styles[i]) for i in range(len(species))])


    with open(os.path.join(flder,"species.txt"),'w') as fl:
        fl.write("\n".join(species))
    with open(os.path.join(flder,"media.txt"),'w') as fl:
        fl.write("{}".format(growth_media))

    initial_abundance = dict([(sp,0.1) for sp in species])


    oxygen_in = {"O2_e0":100}

    with open(os.path.join(flder,"metabolite_inflow.txt"),'w') as fl:
        fl.write("{}".format(oxygen_in))

    with open(os.path.join(flder,"example.log"),'w') as fl:
        metconsin_return = metconsin_sim(species,model_info_fl,initial_abundance = initial_abundance,endtime = et,media = growth_media,metabolite_inflow = oxygen_in, ub_funs = "linear",flobj = fl,resolution = 0.01,ub_params = uptake_params)
                                            


    flder2 = os.path.join(flder,"full_sim")#

    save_metconsin(metconsin_return, flder2)


    #### Prettier plotting:
    font = {'size': 20}

    plt.rc('font', **font)

    cmap = tuple([cdict[sp] for sp in species])

    fig,ax = plt.subplots(figsize = (27,9))
    fltered = metconsin_return["Microbes"].T.iloc[np.linspace(0,metconsin_return["Microbes"].shape[1]-1,23).astype(int)]
    fltered.plot(ax = ax,colormap = ListedColormap(cmap),style=sdict,ms=10)
    metconsin_return["Microbes"].T.plot(ax = ax,colormap = ListedColormap(cmap),legend = False)
    ax.set_xlim(0,0.7)#min(metconsin_return["Microbes"].columns[-1],et))
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Simulated Biomass")
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,20)
    cx = np.arange(0,1,0.1)
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
            stl = ':'
        else:
            col = cdict[chngat[0]]
            stl = sdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,stl,color = col,ms=10)
    plt.savefig(os.path.join(flder,"microbes.png"))
    plt.close()

    ## Plot only the metabolites that changed significantly
    f1 = lambda x: np.any(x>1.1*x[0])
    produced = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f1,axis = 1)]
    f2 = lambda x: np.any(x<0.9*x[0])
    consumed = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f2,axis = 1)]
    reldf = pd.concat([produced,consumed])
    mcldict = dict([(reldf.index[i],plt.cm.jet(i/len(reldf))) for i in range(len(reldf))])

    fig,ax = plt.subplots(figsize = (27,9))
    cmap = tuple([mcldict[met] for met in produced.index])
    produced.T.plot(ax = ax,colormap = ListedColormap(cmap),linewidth = 5,style = ['-',':','-.','--','-o'],ms = 10)#,legend = False)
    ax.set_xlim(0,0.7)#min(metconsin_return["Microbes"].columns[-1],et))
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Simulated Biomass")
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,20)
    cx = np.arange(0,1,0.1)
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
            stl = ':'
        else:
            col = cdict[chngat[0]]
            stl = sdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,stl,color = col,ms=10)
    plt.savefig(os.path.join(flder,"produced_metabolites.png"))
    plt.close()

    fig,ax = plt.subplots(figsize = (27,9))
    cmap = tuple([mcldict[met] for met in consumed.index])
    consumed.T.plot(ax = ax,colormap = ListedColormap(cmap),linewidth = 5,style = ['-',':','-.','--','-o'],ms = 10)#,legend = False)
    ax.set_xlim(0,0.7)#min(metconsin_return["Microbes"].columns[-1],et))
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Simulated Biomass")
    bottom,top = ax.get_ylim()
    yy = np.linspace(bottom,top,20)
    cx = np.arange(0,1,0.1)
    for ti in metconsin_return["BasisChanges"].columns:
        chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
        if len(chngat) > 1 or len(chngat) == 0:
            col = (0,0,0)
            stl = ':'
        else:
            col = cdict[chngat[0]]
            stl = sdict[chngat[0]]
        ax.plot([ti]*len(yy),yy,stl,color = col,ms=10)
    plt.savefig(os.path.join(flder,"consumed_metabolites.png"))
    plt.close()

    ###### Next we can get some information about the species-metabolite network.


    for mic in species:
        microbe_results,microb_combined = ah.make_microbe_table(mic,metconsin_return["SpcMetNetworks"])
        microbe_results.to_csv(os.path.join(flder,"{}_networkinfo.tsv".format(mic)),sep = '\t')


        grth_cos = ah.make_microbe_growthlimiter(mic,metconsin_return["SpcMetNetworks"])
        fig,ax = plt.subplots(figsize = (20,10))
        sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Metabolite",ax=ax)
        ax.set_title("{} Limiting Metabolites".format(mic))
        plt.savefig(os.path.join(flder,"{}_limiting_metabolites.png".format(mic)))
        plt.close()

    all_limiters = []
    for ky in metconsin_return["SpcMetNetworks"].keys():
        df = metconsin_return["SpcMetNetworks"][ky]['edges']
        all_limiters += list(df[df["SourceType"] == "Metabolite"]["Source"])
    all_limiters = np.unique(all_limiters)

    for limi in all_limiters:
        limtab,avg_use = ah.make_limiter_table(limi,metconsin_return["SpcMetNetworks"],species)
        limtab.to_csv(os.path.join(flder,"{}_limiter.tsv".format(limi)),sep = '\t')
        fig,ax = plt.subplots(figsize = (20,10))
        grth_cos = ah.make_limiter_plot(limi,metconsin_return["SpcMetNetworks"])
        sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Model",ax=ax)
        ax.legend(loc=2)
        ax.set_title("{} As Growth Limiter".format(limi))
        plt.savefig(os.path.join(flder,"{}_limiter_plot.png".format(limi)))
        plt.close()

    ##### Finally we look at the metabolite-metabolite network

    metconsin_return["MetMetNetworks"]['Combined']['edges'].sort_values("Variance",ascending=False).head(10).to_latex(os.path.join(flder,"MetMetHighestVarEdges.tex"))

    ### The network making cleans up the names.
    metabolite_list = [met.replace("_e0","").replace("_e","") for met in np.array(metconsin_return["Metabolites"].index)]

    avg_in_degrees, var_in_degrees, in_zeros = ah.node_in_stat_distribution(metabolite_list,metconsin_return["MetMetNetworks"])
    avg_out_degrees, var_out_degrees, in_zeros = ah.node_out_stat_distribution(metabolite_list,metconsin_return["MetMetNetworks"])

    avg_in_degrees.to_csv(os.path.join(flder,"MetMetNodeInAvg.tsv"),sep = '\t')
    var_in_degrees.to_csv(os.path.join(flder,"MetMetNodeInVar.tsv"),sep = '\t')

    avg_out_degrees.to_csv(os.path.join(flder,"MetMetNodeOutAvg.tsv"),sep = '\t')
    var_out_degrees.to_csv(os.path.join(flder,"MetMetNodeOutVar.tsv"),sep = '\t')

    highest_in_var = var_in_degrees.sort_values("SumWeight",ascending = False).head(10)
    highest_in_var.to_latex(os.path.join(flder,"highest_node_in_variance.tex"))
    avg_in_degrees.loc[highest_in_var.index].to_latex(os.path.join(flder,"highest_node_in_var_average.tex"))

    highest_out_var = var_out_degrees.sort_values("SumWeight",ascending = False).head(10)
    highest_out_var.to_latex(os.path.join(flder,"highest_node_out_variance.tex"))
    avg_out_degrees.loc[highest_out_var.index].to_latex(os.path.join(flder,"highest_node_out_var_average.tex"))
