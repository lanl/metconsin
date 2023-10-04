import numpy as np
import cobra as cb
import pandas as pd
import time
import contextlib
import sys
import os
import json
import time
from pathlib import Path


from scipy.interpolate import CubicSpline
from scipy.integrate import trapz

current = os.path.dirname(os.path.abspath(''))
parent = os.path.dirname(current)
sys.path.append(parent)

from metconsin import prep_models as pr
from metconsin import metconsin_sim,save_metconsin,metconsin_environment
from metconsin import analysis_helpers as ah
from metconsin import alternative_dfba as ds


def get_diffs(df1,df2,endtime,resul=10**4):
    
    diffs = pd.DataFrame(columns = ["IntegralDiff","MaxDiff","MaxTime"],index = df1.index)
    for sp in df1.index:
        df1_interp = CubicSpline(df1.columns,df1.loc[sp])
        df2_interp = CubicSpline(df2.columns,df2.loc[sp])
        dif = lambda x: np.abs(df1_interp(x)-df2_interp(x))/np.max(df1_interp(x) + df2_interp(x))

        x_ar = np.linspace(0,endtime,resul)
        
        vallist = [dif(x) for x in x_ar]
        mx = np.max(vallist)
        mxti = x_ar[np.argmax(vallist)]
        intdif = trapz(vallist,x=x_ar)
        
        diffs.loc[sp] = [intdif,mx,mxti]
    return diffs


def compare_nets(net1,net2,nm1,nm2):
    net1c = net1.copy()
    net2c = net2.copy()
    net1c.index = ["##".join(net1c.iloc[i][["Source","Target"]]).replace("_e0","") for i in range(len(net1c))]
    net2c.index = ["##".join(net2c.iloc[i][["Source","Target"]]).replace("_e0","") for i in range(len(net2c))]
    
    shared = set(net1c.index).intersection(net2c.index)
    
    prop_sh1 = len(shared)/len(net1c)
    prop_sh2 = len(shared)/len(net2c)
    
    shared_df = pd.DataFrame(index = shared)
    shared_df[nm1] = net1c.loc[shared]["Weight"]
    shared_df[nm2] = net2c.loc[shared]["Weight"]
    shared_df["Difference"] = net1c.loc[shared]["Weight"] - net2c.loc[shared]["Weight"]
    shared_df["ABS_Difference"] = (net1c.loc[shared]["Weight"] - net2c.loc[shared]["Weight"]).abs()
    shared_df["SameSign"] = np.sign(net1c.loc[shared]["Weight"]*net2c.loc[shared]["Weight"])
    
    return [prop_sh1,prop_sh2,(shared_df["SameSign"] != 1).sum(),shared_df.mean()["ABS_Difference"]], shared_df

if __name__=="__main__":


    model_info_fl = "ModelSeed_info.csv"

    et = 2

    metabolite_id_type = "modelSeedID"

    model_info_fl = "ModelSeed_info.csv"

    species = ['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']


    mednm = "Default"

    growth_media = metconsin_environment(species,model_info_fl,metabolite_id_type = metabolite_id_type)


    with open("exchange_bounds_uniform.json") as fl:
        uptake_params = json.load(fl)


    save_flder = "benchmarking_results"
    Path(save_flder).mkdir(parents=True, exist_ok=True)



    initial_abundance = dict([(sp,0.1) for sp in species])


    oxygen_in = {"O2_e0":100}

    NT = 10

    basic_results = pd.DataFrame(index = range(NT),columns = ["Community_Size","MetConSIN_Time","Direct_Time","MetConSIN_EndTIme","Direct_EndTIme","Average_Microbe_Difference","Average_Metabolite_Difference","Max_Microbe_Diff","Max_Metabolite_Diff"])
    avg_net_comps = pd.DataFrame(columns = ["Proportion MetConSIN Shared","Proportion DirectDFBA Shared","Number Different Sign","Average ABS Difference"])

    for i in range(NT):

        trial_flder = os.path.join(save_flder,"{}".format(i))
        Path(os.path.join(trial_flder,"MetConSIN")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(trial_flder,"DirectDFBA")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(trial_flder,"ComparedNetworks")).mkdir(parents=True, exist_ok=True)


        N = np.random.randint(2,5)
        comm = np.random.choice(species,size=N,replace=False)

        print(comm)

        t0 = time.time()
        with open(os.path.join(trial_flder,"metconsin.log"),'w') as fl:
            metconsin_sol =  metconsin_sim(comm,model_info_fl,initial_abundance = initial_abundance,endtime = et,media = growth_media,metabolite_inflow = oxygen_in, ub_funs = "linear",resolution = 0.01,ub_params = uptake_params,flobj = fl)
        t1 = time.time()

        inters = [inter for inter in metconsin_sol["SpcMetNetworkSummaries"].keys() if inter not in ["Combined","Average","Difference"]]
        net_times = [0.5*(float(s.split("-")[1]) + float(s.split("-")[0])) for s in inters]

        with open(os.path.join(trial_flder,"directdfba.log"),'w') as fl:
            direct_sol = ds.direct_solve(comm,model_info_fl,network_times = net_times,initial_abundance = initial_abundance,endtime = et,media = growth_media,metabolite_inflow = oxygen_in, ub_funs = "linear",resolution = 0.01,ub_params = uptake_params,flobj = fl)
        t2 = time.time()

        save_metconsin(metconsin_sol,os.path.join(trial_flder,"MetConSIN"))
        ds.save_directsolve(direct_sol,os.path.join(trial_flder,"DirectDFBA"))

        final_time = min(direct_sol["Microbes"].columns[-1],metconsin_sol["Microbes"].columns[-1])

        mic_diff = get_diffs(metconsin_sol["Microbes"],direct_sol["Microbes"],final_time)
        met_diff = get_diffs(metconsin_sol["Metabolites"],direct_sol["Metabolites"],final_time)

        mic_diff.to_csv(os.path.join(trial_flder,"MicrobeDiffs.tsv"),sep='\t')
        met_diff.to_csv(os.path.join(trial_flder,"MetaboliteDiffs.tsv"),sep='\t')

        mean_mic_diff = mic_diff.mean()
        mean_met_diff = met_diff.mean()

        max_mic_diff = mic_diff.max()
        max_met_diff = met_diff.max()

        basic_results.loc[i] = [len(comm),t1-t0,t2-t1,metconsin_sol["Microbes"].columns[-1],direct_sol["Microbes"].columns[-1],mean_mic_diff['IntegralDiff'],mean_met_diff['IntegralDiff'],max_mic_diff['MaxDiff'],max_met_diff['MaxDiff']]
    
        nets_comps = pd.DataFrame(columns = ["Proportion MetConSIN Shared","Proportion DirectDFBA Shared","Number Different Sign","Average ABS Difference"])
        for j in range(len(net_times)):
            netcomp,netcompdf = compare_nets(metconsin_sol["SpcMetNetworkSummaries"][inters[j]]['edges'],direct_sol["SpcMetNetworks"][net_times[j]]['edges'],"MetConSIN","DirectDFBA")
            nets_comps.loc[str(net_times[j])] = netcomp
            netcompdf.to_csv(os.path.join(trial_flder,"ComparedNetworks","{:.4f}_network.tsv".format(net_times[j])),sep='\t')

        avnetcomp,avnetcompdf = compare_nets(metconsin_sol["SpcMetNetworkSummaries"]["Average"]['edges'],direct_sol["SpcMetNetworks"]["Average"]['edges'],"MetConSIN","DirectDFBA")
        nets_comps.loc["Average"] = avnetcomp
        avg_net_comps.loc[i] = avnetcomp
        avnetcompdf.to_csv(os.path.join(trial_flder,"ComparedNetworks","average_network.tsv"),sep='\t')

        nets_comps.to_csv(os.path.join(trial_flder,"network_comparison_summary.tsv"),sep='\t')


    basic_results.to_csv(os.path.join(save_flder,"BasicResults.csv"))
    avg_net_comps.to_csv(os.path.join(save_flder,"AverageNetComps.csv"))