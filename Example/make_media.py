import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","metabolic_networks","metconsin"))
import pandas as pd
import numpy as np
import cobra as cb
import contextlib
import json

if __name__=="__main__":


    model_info_fl = "ModelSeed_info.csv"

    species = ['bc1011', 'bc1015', 'bc1003', 'bc1002', 'bc1010', 'bc1008','bc1012', 'bc1016', 'bc1001', 'bc1009']

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

    all_media = pd.DataFrame()

    for modelkey in cobra_models.keys():

        print("==============={}===============".format(modelkey))
        model = cobra_models[modelkey]

        #list all reactions the model claims are exchange.
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#


        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exchng_metabolite_ids = [t[1] for t in exchng_metabolite_ids_wrx]


        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name for metab in exchng_metabolite_ids]


        for met in exchng_metabolite_names:
            rxns = [r for r in model.medium.keys() if met in [m.name for m in model.reactions.get_by_id(r).reactants]]

            if len(rxns):
                all_media.loc[met,modelkey] = np.mean([model.medium[r] for r in rxns])
            else:
                all_media.loc[met,modelkey] = 0

    all_media = 0.01*all_media.fillna(0)

    all_media.max(axis = 1).to_csv("growth_media.tsv",sep='\t')