import sys
import os
import pandas as pd
import numpy as np
import cobra as cb
import contextlib
import json


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


if __name__=="__main__":

    intial_grth = 5

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

    minimal_medias = pd.DataFrame()

    for modelkey in cobra_models.keys():

        print("==============={}===============".format(modelkey))
        model = cobra_models[modelkey]

        #list all reactions the model claims are exchange.
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#


        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exchng_metabolite_ids = [t[1] for t in exchng_metabolite_ids_wrx]
        # ex_rx_di = dict(exchng_metabolite_ids_wrx)


        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name for metab in exchng_metabolite_ids]

        mod_min_med = cb.medium.minimal_medium(model,intial_grth,minimize_components=10)
        # reindx = [ex_rx_di[ky] for ky in mod_min_med.index]
        # mod_min_med.index=reindx
        # minimal_medias[modelkey] = mod_min_med
        # print(mod_min_med.shape)


        for met in exchng_metabolite_names:
            rxns = [r for r in model.medium.keys() if met in [m.name for m in model.reactions.get_by_id(r).reactants]]

            if len(rxns):
                all_media.loc[met,modelkey] = np.mean([model.medium[r] if r in model.medium.keys() else 0 for r in rxns])
                minimal_medias.loc[met,modelkey] = np.mean([mod_min_med.loc[r] if r in mod_min_med.index else 0 for r in rxns])
            else:
                all_media.loc[met,modelkey] = 0
                minimal_medias.loc[met,modelkey] = 0

    all_media = 0.1*all_media.fillna(0)

    all_media.max(axis = 1).to_csv("uniform_media.tsv",sep='\t')

    carbon_sources = ["Galactose_e0","D-Glucose_e0"]
    nitrogen_sources = ["Nitrate_e0","Nitrite_e0","NH3_e0"]

    carbon_limited = all_media.max(axis = 1).copy()
    nitrogen_limited = all_media.max(axis = 1).copy()
    both_limited = all_media.max(axis = 1).copy()

    for carb in carbon_sources:
        carbon_limited.loc[carb] = 0.1*carbon_limited.loc[carb]
        both_limited.loc[carb] = 0.1*both_limited.loc[carb]

    for nit in nitrogen_sources:
        nitrogen_limited[nit] = 0.1*nitrogen_limited.loc[nit]
        both_limited[nit] = 0.1*both_limited.loc[nit]

    carbon_limited.to_csv("carbon_limited_media.tsv",sep = '\t')
    nitrogen_limited.to_csv("nitrogen_limited_media.tsv",sep = '\t')
    both_limited.to_csv("limited_media.tsv",sep = '\t')

    minimal_medias.fillna(0).max(axis=1).to_csv("minimal_media_{}.tsv".format(intial_grth),sep = '\t')
    
