import sys
import os
import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sb

from matplotlib.colors import ListedColormap


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from metconsin import metconsin_sim,save_metconsin
import analysis_helpers as ah

if __name__=="__main__":


    model_info_fl = "ModelSeed_info.csv"

    species1 = ["bc1011","bc1008","bc1016"]#['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']
    species2 = ["bc1011","bc1008","bc1012"]

    growth_media = pd.read_csv("growth_media.tsv",sep = '\t',index_col = 0).squeeze("columns").to_dict()


    with open("exchange_bounds.json") as fl:
        uptake_params = json.load(fl)

    tmlabel = dt.datetime.now()

    pflder = "ComparisonExampleResults_{}".format(tmlabel.strftime("%a%B%d_%Y_%H.%M"))

    # We can change the environment by metabolite ID
    growth_media["D-Glucose_e0"] = 10
    growth_media["O2_e0"] = 10


    Path(pflder).mkdir(parents=True, exist_ok=True)

    oricmap = plt.cm.tab10.colors
    cdict1 = {species1[0]:oricmap[0],species1[1]:oricmap[1],species1[2]:oricmap[2]}
    cdict2 = {species2[0]:oricmap[0],species2[1]:oricmap[1],species2[2]:oricmap[3]}
    mycmap1 = (oricmap[0],oricmap[1],oricmap[2])
    mycmap2 = (oricmap[0],oricmap[1],oricmap[3])
    cdicts = [cdict1,cdict2]
    cmaps = [mycmap1,mycmap2]

    styles = ['o','v','^','>','<','s','P','*','X','D']
    sdict1 = {species1[0]:styles[0],species1[1]:styles[1],species1[2]:styles[2]}
    sdict2 = {species2[0]:styles[0],species2[1]:styles[1],species2[2]:styles[3]}
    sdicts = [sdict1,sdict2]


    for si,species in enumerate([species1,species2]):

        flder = os.path.join(pflder,"{}_{}_{}".format(species[0],species[1],species[2]))
        Path(flder).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(flder,"species.txt"),'w') as fl:
            fl.write("\n".join(species))
        with open(os.path.join(flder,"media.txt"),'w') as fl:
            fl.write("{}".format(growth_media))

        initial_abundance = dict([(sp,0.1) for sp in species])

        with open(os.path.join(flder,"example.log"),'w') as fl:
            metconsin_return = metconsin_sim(species,model_info_fl,initial_abundance = initial_abundance,endtime = 5,media = growth_media, ub_funs = "linear",ub_params = uptake_params,flobj = fl,resolution = 0.01)
                                                    
            
        flder2 = os.path.join(flder,"full_sim")#

        save_metconsin(metconsin_return, flder2)

        #### Prettier plotting:
        font = {'size': 20}

        plt.rc('font', **font)
        
        cdict = cdicts[si]
        sdict = sdicts[si]
        cmap = cmaps[si]

        fig,ax = plt.subplots(figsize = (27,9))
        fltered = metconsin_return["Microbes"].T.iloc[np.linspace(0,metconsin_return["Microbes"].shape[1]-1,23).astype(int)]
        fltered.plot(ax = ax,colormap = ListedColormap(cmap),style=sdict,ms=10)
        metconsin_return["Microbes"].T.plot(ax = ax,colormap = ListedColormap(cmap),legend = False)
        ax.set_xlim(0,5)
        ax.set_ylim(0,1.75)
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
        plt.savefig(os.path.join(flder,"color_coded_microbes.png"))

        fig,ax = plt.subplots(figsize = (30,10))
        f = lambda x: np.any(x>x[0])
        produced = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f,axis = 1)]
        produced.T.plot(ax = ax,colormap = "tab20")#,legend = False)
        ax.set_xlim(0,4)
        bottom,top = ax.get_ylim()
        yy = np.linspace(bottom,top,50)
        cx = np.arange(0,1,0.1)
        for ti in metconsin_return["BasisChanges"].columns:
            chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
            if len(chngat) > 1 or len(chngat) == 0:
                col = (0,0,0)
            else:
                col = cdict[chngat[0]]
            ax.plot([ti]*len(yy),yy,"o",color = col)
        plt.savefig(os.path.join(flder,"produced_metabolites.png"))

        fig,ax = plt.subplots(figsize = (30,10))
        f = lambda x: np.any(x<0.8*x[0])
        consumed = metconsin_return["Metabolites"][metconsin_return["Metabolites"].apply(f,axis = 1)]
        consumed.T.plot(ax = ax,colormap = "tab20")#,legend = False)
        ax.set_xlim(0,4)
        bottom,top = ax.get_ylim()
        yy = np.linspace(bottom,top,50)
        cx = np.arange(0,1,0.1)

        for ti in metconsin_return["BasisChanges"].columns:
            chngat = metconsin_return["BasisChanges"][metconsin_return["BasisChanges"][ti]].index
            if len(chngat) > 1 or len(chngat) == 0:
                col = (0,0,0)
            else:
                col = cdict[chngat[0]]
            ax.plot([ti]*len(yy),yy,"o",color = col)
        plt.savefig(os.path.join(flder,"consumed_metabolites.png"))


        ###### Next we can get some information about the species-metabolite network.


        for mic in species:
            microbe_results = ah.make_microbe_table(mic,metconsin_return["SpcMetNetworks"])
            microbe_results.to_csv(os.path.join(flder,"{}_networkinfo.tsv".format(mic)),sep = '\t')
            grth_cos = ah.make_microbe_growthlimiter(mic,metconsin_return["SpcMetNetworks"])
            fig,ax = plt.subplots(figsize = (20,10))
            sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Metabolite",ax=ax)
            ax.set_title("{} Limiting Metabolites".format(mic))
            plt.savefig(os.path.join(flder,"{}_limiting_metabolites.png".format(mic)))

        all_limiters = []
        for ky in metconsin_return["SpcMetNetworks"].keys():
            df = metconsin_return["SpcMetNetworks"][ky]['edges']
            all_limiters += list(df[df["SourceType"] == "Metabolite"]["Source"])
        all_limiters = np.unique(all_limiters)

        for limi in all_limiters:
            limtab = ah.make_limiter_table(limi,metconsin_return["SpcMetNetworks"],species)
            limtab.to_csv(os.path.join(flder,"{}_limiter.csv".format(limi)),sep = '\t')
            fig,ax = plt.subplots(figsize = (20,10))
            grth_cos = ah.make_limiter_plot(limi,metconsin_return["SpcMetNetworks"])
            sb.barplot(data = grth_cos,y = "Coefficient",x = "TimeRange",hue = "Model",ax=ax)
            ax.legend(loc=2)
            ax.set_title("{} As Growth Limiter".format(limi))
            plt.savefig(os.path.join(flder,"{}_limiter_plot.png".format(limi)))


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