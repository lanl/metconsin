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

    metabolite_id_type = "modelSeedID"
    metabolite_tag = "_e0"

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

    all_medias = pd.DataFrame()
    met_ids = pd.Series(dtype=str)

    for modelkey in cobra_models.keys():

        model = cobra_models[modelkey]

        #list all reactions the model claims are exchange.
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#


        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exchng_metabolite_ids = [t[1] for t in exchng_metabolite_ids_wrx]

        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name for metab in exchng_metabolite_ids]

        for mi,met in enumerate(exchng_metabolite_names):

            if met not in met_ids.index:
                met_ids.loc[met] = exchng_metabolite_ids[mi]

            rxns = [r for r in model.medium.keys() if met in [m.name for m in model.reactions.get_by_id(r).reactants]]
            if len(rxns):
                all_medias.loc[met,modelkey] = np.mean([model.medium[r] if r in model.medium.keys() else 0 for r in rxns])
            else:
                all_medias.loc[met,modelkey] = 0

    intitial_media = 0.1*all_medias.fillna(0).max(axis = 1)
    media = intitial_media.copy()

    # print(met_ids)

    agora_media_nm = sys.argv[1]

    if "{}_AGORA.tsv".format(agora_media_nm) in os.listdir(agora_media_loc):
        agora_media_nm = "{}_AGORA.tsv".format(agora_media_nm)

    if agora_media_nm in os.listdir(agora_media_loc):
        agora_media = pd.read_csv(os.path.join(agora_media_loc,agora_media_nm),index_col = 0, sep = '\t')

    else:
        print("Cannot find media file {}".format(os.path.join(agora_media_loc,agora_media_nm)))
        sys.exit()

    for met in intitial_media.index:
        metid = met_ids.loc[met].replace(metabolite_tag,"")
        if metid in agora_media[metabolite_id_type].values:
            media.loc[met] = agora_media[agora_media[metabolite_id_type] == metid]["fluxValue"].iloc[0]

    media.to_csv("media_{}".format(agora_media_nm),sep = '\t')

