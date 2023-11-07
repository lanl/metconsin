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

import itertools

from matplotlib.colors import ListedColormap

from itertools import combinations as com

current = os.path.dirname(os.path.abspath(''))
parent = os.path.dirname(current)
metcon = os.path.dirname(parent)
sys.path.append(metcon)

import metconsin
from metconsin import analysis_helpers as ah

if __name__=="__main__":


    model_info_flnm = "weiss_model_info.csv"
    weiss_medium_fl = "weiss_medium.csv"


    file_info = pd.read_csv(model_info_flnm,index_col = 0)
    spc = file_info["Species"]

    env = metconsin.metconsin_environment(spc,model_info_flnm,media_source = weiss_medium_fl,metabolite_id_type="metabolite")

    save_flder = 'weiss_results'
    Path(os.path.join(save_flder,"full_community")).mkdir(parents=True, exist_ok=True)

    initial_abundance = dict([(sp,0.1) for sp in spc])
    oxygen_in = {"O2-e0":100}

    et = 20

    with open("weiss_full_community.log",'w') as fl:
        full_community = metconsin.metconsin_sim(spc,model_info_flnm,media = env,initial_abundance = initial_abundance,metabolite_inflow = oxygen_in,endtime = et,flobj = fl)

    metconsin.save_metconsin(full_community, os.path.join(save_flder,"full_community"))

    for sp in spc:
        Path(os.path.join(save_flder,"{}_results".format(sp))).mkdir(parents=True, exist_ok=True)
        with open("{}.log".format(sp),'w') as fl:
            monosim = metconsin.metconsin_sim([sp],model_info_flnm,media = env,initial_abundance = initial_abundance,metabolite_inflow = oxygen_in,endtime = et,flobj = fl) 
        metconsin.save_metconsin(monosim,os.path.join(save_flder,"{}_results".format(sp)))

    for pair in itertools.combinations(spc,2):
        fld = os.path.join(save_flder,"{}_{}_results".format(pair[0],pair[1]))
        Path(fld).mkdir(parents=True, exist_ok=True)
        with open("{}_{}.log".format(pair[0],pair[1]), 'w') as fl:
            pairsim =  metconsin.metconsin_sim(pair,model_info_flnm,media = env,initial_abundance = initial_abundance,metabolite_inflow = oxygen_in,endtime = et,flobj = fl) 
        metconsin.save_metconsin(pairsim,fld)
