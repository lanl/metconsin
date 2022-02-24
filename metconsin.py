from findwaves import *
from make_network import *
import time
import pandas as pd

def metconsin(desired_models,model_info_file,media = {},uptake_dicts = {},random_kappas="ones"):
    '''

    NOTE: as of now uptake is always linear.
    uptake_dicts => can give uptake parameters for each microbe/metabolite pair
    random_kappas => how to choose non-given uptake parameters

    '''
    start_time = time.time()

    cobra_models = {}

    model_info = pd.read_csv(model_info_file)

    for mod in desired_models:
        if any(model_info.Species == mod):
            flnm = model_info.loc[model_info.Species == mod,'File'].iloc[0]
            cobra_models[mod] = cb.io.load_json_model(flnm)
            if not cobra_models[mod].name:
                cobra_models[mod].name = mod
        else:
            print("Error: No model of species " + mod)


    print("Loaded " + str(len(cobra_models)) + " models successfully")


    for model in cobra_models:
        if model in media.keys():
            if isinstance(media[model],dict):
                tmp_medium = {}
                exc = [rxn.id for rxn in cobra_models[model].reactions if 'EX_' in rxn.id]
                for ky in media[model].keys():
                    if ky in exc:
                        tmp_medium[ky] = media[model][ky]
                cobra_models[model].medium = tmp_medium

            elif media[model] == "minimal":
                mxg = cobra_models[model].slim_optimize()
                min_med = cb.medium.minimal_medium(cobra_models[model],mxg,minimize_components = True)
                cobra_models[model].medium = min_med
                cobra_models[model].medium = min_med


    #returns dict of surfmods, list of metabolites, and concentration of metabolites.
    models,mets,mets0 = prep_cobrapy_models(cobra_models,uptake_dicts = uptake_dicts ,random_kappas=random_kappas)
    #for new network after perturbing metabolites, we only need to update mets0.
    #mets establishes an ordering of metabolites.
    #Next establish an ordering of microbes. Note, this won't necessarily be predictable, python
    #dict keys are unordered and calling dict.keys() will give whatever ordering it wants.
    model_order = list(models.keys())

    bases = get_waves(models,mets,mets0)

    minuts,sec = divmod(time.time() - start_time, 60)
    # try:
    #     flobj.write("prep_indv_model", self.Name,": Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    # except:
    print("[MetConSIN] Bases computed in ",int(minuts)," minutes, ",sec," seconds.")

    smi_nodes,cof_edges,sum_edges = species_metabolite_network(bases,mets,mets0,models)

    return {"Nodes":smi_nodes,"FullEdgeSet":cof_edges,"SummaryEdgeSet":sum_edges,"ModelList":model_order, "SurfModels":models, "Metabolites":mets, "MetabConcentrations":mets0, "Bases":bases}
