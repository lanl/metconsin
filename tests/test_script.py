import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","metconsin"))
from metconsin import *
import pandas as pd
from pathlib import Path
import datetime as dt

if __name__=="__main__":
    species = ["S.cerevisiae","M.tuberculosis","E.coli","P.putida"]
    # modellist, models, metlist, metcons, bases = metconsin(species,"bigg_model_file_info.txt",media ={"S.cerevisiae":"minimal","P.putida":"minimal","M.tuberculosis":"minimal"})
    # smi_nodes,cof_edges,sum_edges = species_metabolite_network(bases,metlist,metcons,models)

    species_metabolite_interaction_network = metconsin(species,"bigg_model_file_info.txt",media ={"S.cerevisiae":"minimal","P.putida":"minimal","M.tuberculosis":"minimal"})

    x = dt.datetime.now()

    flder = os.path.join(os.path.expanduser("~"),"Documents","metabolic_network","testNetwork"+x.strftime("%a%d%H%M"))

    Path(flder).mkdir(parents=True, exist_ok=True)



    species_metabolite_interaction_network["Nodes"].to_csv(os.path.join(flder,"smi_nodes.tsv"),sep="\t")
    species_metabolite_interaction_network["FullEdgeSet"].to_csv(os.path.join(flder,"cof_edges.tsv"),sep="\t")
    species_metabolite_interaction_network["SummaryEdgeSet"].to_csv(os.path.join(flder,"sum_edges.tsv"),sep="\t")
