import requests as rq 
import pandas as pd
import numpy as np
import json
import sys


def get_AGORA_diets():

    """
    Function to download all the available diet files from `AGORA <https://www.vmh.life/#nutrition>`_

    Saves diet metadata as a .json

    Saves .tsv with reaction fluxes, metabolite names, and metabolite ID translations to KEGG, BIGG, PubChem, and ModelSEED for each file

    :return: None


    """

    diet_list = rq.get("https://www.vmh.life/_api/diets/")

    with open("diet_list.json",'w') as fl:
        json.dump(diet_list.json()['results'],fl)

    # print(diet_list.json()['results'][0].keys())

    diet_dfs = {}

    for df in diet_list.json()['results']:

        print(df['name'])

        diet = rq.get("https://www.vmh.life/_api/dietflux/?diet={}&page_size=1000".format(df['name']))

        diet_json = diet.json()['results']


        diet_dict = dict([(ent["metabolite"],ent) for ent in diet_json])

        diet_dfs[df['name']] = pd.DataFrame.from_dict(diet_dict,orient='index')


    all_diets = pd.concat(list(diet_dfs.values()))

    all_metabolites = np.unique(all_diets.index)


    for metab in all_metabolites:

        print(metab)

        matab_info = rq.get("https://www.vmh.life/_api/metabolites/?abbreviation={}".format(metab))

        all_diets.loc[metab,"fullName"] = matab_info.json()['results'][0]['fullName']
        all_diets.loc[metab,"biggID"] = matab_info.json()['results'][0]['biggId']
        all_diets.loc[metab,"keggID"] = matab_info.json()['results'][0]['keggId']
        all_diets.loc[metab,"pubChemId"] = matab_info.json()['results'][0]['pubChemId']
        all_diets.loc[metab,"modelSeedID"] = matab_info.json()['results'][0]['seed']
        all_diets.loc[metab,"miriam"] = matab_info.json()['results'][0]['miriam']


    for diet_nm in np.unique(all_diets['diet']):

        dietdf = all_diets[all_diets['diet'] == diet_nm]

        svnm = diet_nm.replace(",","").replace(" ","_")

        dietdf.to_csv("{}_AGORA.tsv".format(svnm),sep = '\t')

if __name__ == "__main__":

    get_AGORA_diets()
