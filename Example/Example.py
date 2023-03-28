import sys
import os
import path
import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import cobra as cb
import contextlib
import json

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from metconsin import metconsin_sim,save_metconsin

if __name__=="__main__":


    model_info_fl = "ModelSeed_info.csv"

    species = ['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']

    growth_media = pd.read_csv("growth_media.tsv",sep = '\t',index_col = 0).squeeze("columns").to_dict()


    with open("exchange_bounds.json") as fl:
        uptake_params = json.load(fl)

    tmlabel = dt.datetime.now()

    flder = "modelSeed_{}s_{}".format(len(species),tmlabel.strftime("%a%B%d_%Y_%H.%M"))



    Path(flder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(flder,"species.txt"),'w') as fl:
        fl.write("\n".join(species))
    with open(os.path.join(flder,"media.txt"),'w') as fl:
        fl.write("{}".format(growth_media))



    with open("example.log",'w') as fl:
        metconsin_return = metconsin_sim(species,model_info_fl,endtime = 2,media = growth_media, ub_funs = "linear",ub_params = uptake_params,flobj = fl,resolution = 0.01)
                                                
        
    flder = os.path.join(flder,"full_sim")#

    save_metconsin(metconsin_return, flder)
