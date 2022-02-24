import numpy as np
import scipy as sp
try:
    import cplex as cp
except ImportError:
    pass

try:
    import gurobipy as gb
except ImportError:
    pass

try:
    import cylp as coin
    from cylp.cy import CyClpSimplex
    from cylp.py.modeling.CyLPModel import CyLPArray
except ImportError:
    pass

import pandas as pd
import time
import cobra as cb

class SurfMod:
    def __init__(self,G1,G2,lilg,ilbs,iubs,kaps,elbs,exchanged_metabolites,Name = None,deathrate = 0):

        G10 = np.array(G1)
        self.Gamma1 = np.concatenate([G10,-G10],axis = 1)#Exchange rows (GammaStar)
        G20 = np.array(G2)
        self.Gamma2 = np.concatenate([G20,-G20],axis = 1)#Internal (GammaDagger)
        ob0 = np.array(lilg)
        self.objective = np.concatenate([lilg,-lilg])
        self.intLB = np.array(ilbs)
        self.intUB = np.array(iubs)
        self.uptakes = np.array(kaps)
        self.exchgLB = np.array(elbs)
        if Name == None:
            self.Name = ''.join([str(np.random.choice(list('abcdefghijklmnopqrstuvwxyz123456789'))) for n in range(5)])
        else:
            self.Name = Name
        A0 = np.concatenate([-np.array(G1),np.array(G1),np.eye(np.array(G1).shape[1]),-np.eye(np.array(G1).shape[1])],axis = 0)
        A1 = np.concatenate([A0,-A0],axis = 1)
        self.MatrixA = np.concatenate([A1,-np.eye(A1.shape[1])],axis = 0)
        self.statbds = np.concatenate([-np.array(elbs),np.array(iubs),-np.array(ilbs),np.zeros(A1.shape[1])])#np.empty(0)
        self.deathrate = deathrate
        self.exchanged_metabolites = exchanged_metabolites

    def prep_indv_model(self,initial_N,secondobj = [],report_activity = True, solver = 'coin',flobj = None):

        Gamma1 = self.Gamma1
        Gamma2 = self.Gamma2
        obje = self.objective
        low_int = self.intLB
        up_int = self.intUB
        alphas = self.uptakes
        low_exch = self.exchgLB
        MatrixA = self.MatrixA
        statbds = self.statbds



        t1 = time.time()
        Gamma1 = Gamma1.astype(float)
        Gamma2 = Gamma2.astype(float)


        #All that matters is Ker(Gamma2), so we can replace Gamma2 with a new
        #matrix with orthonormal rows that has the same kernal.

        Z = sp.linalg.null_space(Gamma2)
        Gamma2tT = sp.linalg.null_space(Z.T)
        Gamma2 = Gamma2tT.T

        self.Gamma2 = Gamma2

        obje = obje.astype(float)
        low_int = low_int.astype(float)
        up_int = up_int.astype(float)####Should check to make sure all LB <= UB
        alphas = alphas.astype(float)
        low_exch = np.minimum(low_exch,alphas*initial_N)


        upbds_exch = initial_N*alphas

        if report_activity:
            try:
                flobj.write("prep_indv_model: initializing LP\n")
            except:
                print("prep_indv_model: initializing LP")
        growth = gb.Model("growth")
        growth.setParam( 'OutputFlag', False )


        sparms = [growth.addVar(lb = - gb.GRB.INFINITY,ub = gb.GRB.INFINITY, name = "s" + str(i)) for i in range(MatrixA.shape[1])]
        growth.update()
        objv = gb.quicksum([a[0]*a[1] for a in zip(obje,sparms)])
        growth.setObjective(objv,gb.GRB.MAXIMIZE)

        bds_vec = np.concatenate([upbds_exch,statbds])
        if report_activity:
            try:
                flobj.write("prep_indv_model: Adding constraints\n")
            except:
                print("prep_indv_model: Adding constraints")


        growth.addConstrs((gb.quicksum([MatrixA[i][l]*sparms[l] for l in range(len(sparms))]) <= bds_vec[i] for i in range(len(MatrixA))), name = 'LE')
        growth.addConstrs((gb.quicksum([Gamma2[i][l]*sparms[l] for l in range(len(sparms))]) == 0 for i in range(len(Gamma2))), name = 'Kernal')
        growth.update()

        if report_activity:
            try:
                flobj.write("prep_indv_model: optimizing LP\n")
                flobj.write("prep_indv_model: optimizing with " + str(len(growth.getConstrs())) + " constraints\n" )
            except:
                print("prep_indv_model: optimizing LP")
                print("prep_indv_model: optimizing with ",len(growth.getConstrs()) ," constraints" )
        growth.optimize()


        status = growth.status
        # if status == 2:
        #
        # print(status)

        statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
        if status in statusdic.keys():
            if report_activity:
                try:
                    flobj.write("prep_indv_model: LP Status: " +  statusdic[status] + '\n')
                except:
                    print("prep_indv_model: LP Status: ", statusdic[status])
        else:
            if report_activity:
                try:
                    flobj.write("prep_indv_model: LP Status: Other\n")
                except:
                    print("prep_indv_model: LP Status: Other")

        if status == 2:


            # wi = np.array([v.x for v in growth.getVars()])#growth.solution.get_values()
            val = growth.objVal

            if len(secondobj) != len(sparms):#if not given a valid second objective, minimize total flux
                secondobj = -np.ones(len(sparms))


            growth.addConstr(objv == val)
            growth.update()
            newobj = gb.quicksum([a[0]*a[1] for a in zip(secondobj,sparms)])
            growth.setObjective(newobj,gb.GRB.MAXIMIZE)
            growth.update()
            growth.optimize()

            wi = np.array([v.x for v in growth.getVars()])



            # static2 = np.concatenate([-low_exch,up_int,-low_int])
            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write("prep_indv_model: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print("prep_indv_model: Done in ",int(minuts)," minutes, ",sec," seconds.")


            # self.statbds = static2
            return wi#,(MatrixA,static2,alphas,Gamma1,Gamma2,obje,death)
        else:
            return np.array(["failed to prep"])



def prep_cobrapy_models(models,uptake_dicts = {},extracell = 'e', random_kappas = "new",media = {}):


    #can provide metabolite uptake dictionary as dict of dicts {model_key1:{metabolite1:val,metabolite2:val}}

    from cobra import util

    if not isinstance(models,dict):
        modeldict = {}
        for mod in models:
            modeldict[mod.name] = mod
        models = modeldict

    if len(media):
        for mod in models:
            models[mod].medium = media

    metaabs = {}
    y0s = {}
    exrn = {}
    metabids = {}
    nametoid = {}
    nametorxnid = {}
    urts = {}

    rand_str_loc = 0

    for modelkey in models.keys():
        model = models[modelkey]


        if modelkey not in uptake_dicts.keys():
            uptake_dicts[modelkey] = {}

        # try:
        #     exchng_reactions = list(model_meds[modelkey].keys())
        # except:
        exchng_reactions = [rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#list(model.medium.keys())


        exchng_metabolite_ids = [metab.id for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #


        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name for metab in exchng_metabolite_ids]


        nutrient_concentrations = {}



        if len(uptake_dicts[modelkey]) < len(exchng_reactions):#ones or random numbers
            try:
                random_nums = np.load(random_kappas)
                loadedrand = True
            except:
                random_nums = np.empty(0)
                loadedrand = False
                if random_kappas == "ones":
                    print("prep_cobrapy_models: Will use uniform uptake parameters = 1")
                else:
                    print("prep_cobrapy_models: Will create random uptake")

            if loadedrand:
                if rand_str_loc < len(random_nums):
                    uptake_rate1 = random_nums[rand_str_loc:(rand_str_loc + len(exchng_reactions))]
                    rand_str_loc = rand_str_loc + len(exchng_reactions)
                    uptkdict1 = dict(zip(exchng_metabolite_names,uptake_rate1))
                else:
                    random_nums = np.concatenate([random_nums,np.random.rand(len(exchng_reactions))])
                    uptake_rate1 = random_nums[rand_str_loc:(rand_str_loc + len(exchng_reactions))]
                    rand_str_loc = rand_str_loc + len(exchng_reactions)
                    uptkdict1 = dict(zip(exchng_metabolite_names,uptake_rate1))
            else:
                if random_kappas == "ones":
                    random_nums = np.concatenate([random_nums,np.ones(rand_str_loc+len(exchng_reactions))])
                    uptake_rate1 = random_nums[rand_str_loc:(rand_str_loc + len(exchng_reactions))]
                    rand_str_loc = rand_str_loc + len(exchng_reactions)
                    uptkdict1 = dict(zip(exchng_metabolite_names,uptake_rate1))
                else:
                    random_nums = np.concatenate([random_nums,np.random.rand(rand_str_loc+len(exchng_reactions))])
                    uptake_rate1 = random_nums[rand_str_loc:(rand_str_loc + len(exchng_reactions))]
                    rand_str_loc = rand_str_loc + len(exchng_reactions)
                    uptkdict1 = dict(zip(exchng_metabolite_names,uptake_rate1))

            uptkdict = {}
            for ky in uptkdict1.keys():
                if ky in uptake_dicts[modelkey].keys():
                    uptkdict[ky] = uptake_dicts[modelkey][ky]
                else:
                    uptkdict[ky] = uptkdict1[ky]


        else:
            ##Translate uptake dicts from reaction id to metabolite name.
            uptkdict = {}
            for rxn in uptake_dicts[modelkey]:
                metabs = [metab.name for metab in model.reactions.get_by_id(rxn).reactants]
                for met in metabs:
                    uptkdict[met] = uptake_dicts[modelkey][rxn]

        # print(uptkdict)
        uptake_rate = [uptkdict[met] for met in exchng_metabolite_names]

        i = 0


        for er in exchng_reactions:
            al = uptake_rate[i]
            i += 1
            if er in model.medium.keys():
                nutrient_concentrations[er] = model.medium[er]/(al)
            else:
                nutrient_concentrations[er] = 0
            # uptake_rate+= [al]




        i = 0
        nmid = dict(zip(exchng_metabolite_ids,exchng_metabolite_names))
        idnm = dict(zip(exchng_metabolite_names,exchng_metabolite_ids))
        rxnid = dict(zip(exchng_reactions,exchng_metabolite_ids))
        nmtorxn = dict(zip(exchng_metabolite_names,exchng_reactions))
        y_init = dict([(nmid[rxnid[ky]],nutrient_concentrations[ky]) for ky in nutrient_concentrations.keys()])

        metaabs[model.name] = exchng_metabolite_names
        y0s[model.name] = y_init
        exrn[model.name] = exchng_reactions
        metabids[model.name] = exchng_metabolite_ids
        nametoid[model.name] = idnm
        nametorxnid[model.name] = nmtorxn
        urts[model.name] = uptkdict



    ##### NOW: we have to reconcile the exchanged metabolites. Swapping order means swapping rows of Gamma1! So
    ### we must agree on an order.
    masterlist = []
    for li in metaabs.values():
        masterlist += li
    masterlist = np.unique(masterlist)



    #### Initial y is not as difficult. Average them out.
    mastery0 = {}
    for nm in masterlist:
        yyy0 = 0
        ctt = 0
        for mod in y0s.values():
            if nm in mod.keys():
                yyy0 += mod[nm]
                ctt += 1
        if ctt:
            mastery0[nm] = yyy0/ctt
        else:
            mastery0[nm] = 0





    real_model = {}
    namemap ={}
    for modelkey in models.keys():
        model = models[modelkey]

        #Get the stoichiometric matrix and break it apart
        ###Index is metabolite ID, columns are rxn ID
        Gamma = util.array.create_stoichiometric_matrix(model, array_type = 'DataFrame')


        for meta in masterlist:
            if meta not in metaabs[model.name]:
                blnk = pd.DataFrame([np.zeros(len(Gamma.columns))],columns = Gamma.columns, index = [meta])
                Gamma = Gamma.append(blnk)
                # print(meta)
            elif nametoid[model.name][meta] not in Gamma.index:
                blnk = pd.DataFrame([np.zeros(len(Gamma.columns))],columns = Gamma.columns, index = [nametoid[model.name][meta]])
                Gamma = Gamma.append(blnk)


        mastertoids = [nametoid[model.name][nm] if nm in nametoid[model.name].keys() else nm for nm in masterlist]



        internal_reactions = np.array(Gamma.columns)[[rxn not in exrn[model.name] for rxn in Gamma.columns]]


        internal_metabs = np.array(Gamma.index)[[((met not in metabids[model.name]) and (met not in masterlist)) for met in Gamma.index]]

        EyE = Gamma.loc[np.array(metabids[model.name]),np.array(exrn[model.name])]




        if (-EyE.values == np.eye(EyE.values.shape[0])).all():
            Gamma1 = Gamma.loc[np.array(mastertoids),internal_reactions]
        else:
            Gamma1 = -Gamma.loc[np.array(mastertoids),internal_reactions]


        # =============================================================================

        Gamma2 = Gamma.loc[internal_metabs,internal_reactions]
        Gamma1ar = Gamma1.values
        Gamma2ar = Gamma2.values


        #Next we need the objective function that identifies growth - the flux that COBRA optimizes
        growth_col = pd.Series(np.zeros(len(internal_reactions)),index = Gamma2.columns)
        biom_rxns = [rxn.id for rxn in util.solver.linear_reaction_coefficients(model).keys()]
        growth_col.loc[biom_rxns] = 1
        lilgamma = growth_col.values


        real_reactions = [ky for ky in nametorxnid[model.name].keys() if model.reactions.has_id(nametorxnid[model.name][ky])]


        exchng_lower_bounds = np.array([-model.reactions.get_by_id(nametorxnid[model.name][nm]).bounds[1] if nm in real_reactions else 0 for nm in masterlist])

        internal_upper_bounds = np.array([rxn.bounds[1] for rxn in model.reactions if rxn.id not in exrn[model.name]])
        internal_lower_bounds = np.array([min(0,rxn.bounds[0]) for rxn in model.reactions if rxn.id not in exrn[model.name]])


        kappas = np.array([urts[model.name][nm] if nm in urts[model.name].keys() else 0 for nm in masterlist])

        real_model[modelkey] = SurfMod(Gamma1ar,Gamma2ar,lilgamma,internal_lower_bounds,internal_upper_bounds,kappas,exchng_lower_bounds,metaabs[model.name],Name = model.name)
        #[Gamma1ar,Gamma2ar,lilgamma,internal_lower_bounds,internal_upper_bounds,kappas,exchng_lower_bounds]
        # namemap[modelkey] = model.name


    return real_model,masterlist,mastery0
