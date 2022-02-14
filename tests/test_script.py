import sys
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","metconsin"))
from metconsin import *
import pandas as pd


if __name__=="__main__":
    species = ["S.cerevisiae","M.tuberculosis","E.coli" "P.putida"]
    modellist, models, metlist, metcons, bases = metconsin(species,"bigg_model_file_info.txt",media ={"S.cerevisiae":sc_med,"P.putida":pp_med,"M.tuberculosis":mt_med})
    sm_nodes,sm_edges = species_metabolite_network(bases,metlist,models)
    sm_nodes.to_csv(os.path.join((os.path.expanduse("~"),"Documents","metabolic_network","testNetwork","sm_nodes.csv"))
    sm_edges.to_csv(os.path.join((os.path.expanduse("~"),"Documents","metabolic_network","testNetwork","sm_edges.csv"))
