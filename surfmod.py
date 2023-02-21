from operator import index
from tracemalloc import start
import numpy as np
import scipy.linalg as la
import time
from numba import jit

import sys

class SurfMod:
    def __init__(self,exchanged_metabolites,gamStar,gamDag,objective,intrn_order,interior_lbs,interior_ubs,exterior_lbfuns,exterior_ubfuns,exterior_lbfuns_derivative,exterior_ubfuns_derivatives,lbfuntype = "",ubfuntype = "",Name = None,deathrate = 0,gamma_star_indices = "Auto",forcedOns = True):

        #put the model in standard form by forming the system
        # [GamStar -GamStar I 0 0 0 0 0 0]         [f(y)]
        # [-GamStar GamStar 0 I 0 0 0 0 0] [v_f]   [g(y)]
        # [I         0      0 0 I 0 0 0 0] [v_r] = [interior_ubs]
        # [0         I      0 0 0 I 0 0 0] [s]     [interior_lbs]
        # [GamDag  -GamDag  0 0 0 0 I 0 0]         [0]
        # [-GamDag GamDag   0 0 0 0 0 I 0]         [0]   
        # [    Forced On    0 0 0 0 0 0 I]         [ON]

        # where f,g are the upper/lower exchange bounds, and are functions of the
        #environmental metabolites.

        #All that matters is Ker(GammaDagger), so we can replace GammaDagger with a new
        #matrix with orthonormal rows that has the same kernal.

        self.ezero = 10**-7

        if Name == None:
            self.Name = ''.join([str(np.random.choice(list('abcdefghijklmnopqrstuvwxyz123456789'))) for n in range(5)])
        else:
            self.Name = Name

        Z = la.null_space(gamDag.astype(float))
        gamDagtT = la.null_space(Z.T)
        gamDag = gamDagtT.T

        num_exch,num_v = gamStar.shape
        num_internal = gamDag.shape[0]

        self.GammaStar = np.concatenate([gamStar.astype(float), -gamStar.astype(float)],axis = 1)

        if isinstance(gamma_star_indices,str):
            if gamma_star_indices == "Auto":
                self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)
            else:
                print("Cannot understand exchange mapping. Assuming 1:1 and in order, using default (Auto)")
                self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)
        elif isinstance(gamma_star_indices,np.ndarray):#should add a check here.
            self.ExchangeOrder = gamma_star_indices
        else:
            print("Cannot understand exchange mapping. Assuming 1:1 and in order, using default (Auto)")
            self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)


        self.GammaDagger = np.concatenate([gamDag.astype(float), -gamDag.astype(float)],axis = 1)#gamDag.astype(float)
        self.num_fluxes = 2*num_v
        self.num_exch_rxns = num_exch
        self.num_internal_metabolites = num_internal

        # self.total_var = 6*num_v + 2*num_exch
        rw1 = self.GammaStar#, np.eye(num_exch),np.zeros((num_exch,num_exch+2*num_v))], axis = 1)
        rw2 = -self.GammaStar#np.concatenate([-gamStar.astype(float), gamStar.astype(float)],axis = 1)#, np.zeros((num_exch,num_exch)),np.eye(num_exch),np.zeros((num_exch,2*num_v))], axis = 1)
        rw3 = np.concatenate([np.eye(num_v),np.zeros((num_v,num_v))],axis = 1)#,np.zeros((num_v,num_v+2*num_exch)),np.eye(num_v),np.zeros((num_v,num_v))], axis = 1)
        rw4 = np.concatenate([np.zeros((num_v,num_v)), np.eye(num_v)],axis = 1)#, np.zeros((num_v,2*num_exch+num_v)),np.eye(num_v)], axis = 1)
        rw5 = self.GammaDagger #np.concatenate([gamDag, -gamDag],axis = 1)#, np.zeros((num_internal,2*num_exch+2*num_v))], axis = 1)
        rw6 = -self.GammaDagger #np.concatenate([-gamDag, gamDag],axis = 1)
        prob_mat = np.concatenate([rw1,rw2,rw3,rw4,rw5,rw6], axis = 0)
        

        # self.expandGammaStar = np.concatenate([gamStar.astype(float),-gamStar.astype(float),np.zeros((num_exch,2*num_v + 2*num_exch))],axis = 1)

        self.objective = np.concatenate([-np.array(objective).astype(float),np.array(objective).astype(float)])

        self.flux_order = intrn_order

        self.exchange_bounds = np.concatenate([exterior_ubfuns,exterior_lbfuns])
        self.lower_exch_type = lbfuntype
        self.upper_exch_type = ubfuntype


        interior_ubs_min0 = np.array([max(bd,0) for bd in interior_ubs])
        interior_lbs_min0 = np.array([max(bd,0) for bd in interior_lbs])
        all_internal = np.concatenate([interior_ubs_min0.astype(float),interior_lbs_min0.astype(float),np.zeros(2*num_internal).astype(float)])

        if forcedOns:
        #We need to add constraints for negative bounds.
        ##A negative lower bound means the FORWARD reaction must be above -bd and REVERSE reaction must be 0
        ###         bd>val is becomes -bd<-val in standard form.
        ##A negative upper bound means the REVERSE reaction must be above -bd abd FORWARD reaction must be 0
        ###         bd>val is becomes -bd<-val in standard form.


            fon = []
            foff = []

            neg_lower = np.where(interior_lbs < 0) #need to add a constraint for the FORWARD reaction
            num_lower = len(neg_lower[0])
            if num_lower:
                #Force on Forward
                fon += [(neg_lower[0][p],len(prob_mat) + p) for p in range(num_lower)]
                new_rows = np.concatenate([-np.eye(num_v)[neg_lower],np.zeros((num_lower,num_v))],axis = 1)
                prob_mat = np.concatenate([prob_mat,new_rows],axis = 0)
                #for the constraint a<x<b, need a<b...make sure a = min(a,b)
                all_internal = np.concatenate([all_internal,-np.minimum(-interior_lbs[neg_lower],interior_ubs_min0[neg_lower])])
                #Force off Reverse
                foff += [(neg_lower[0][p] + len(interior_ubs),len(prob_mat) + p) for p in range(num_lower)]
                new_rows = np.concatenate([np.zeros((num_lower,num_v)),np.eye(num_v)[neg_lower]],axis = 1)
                prob_mat = np.concatenate([prob_mat,new_rows],axis = 0)
                all_internal = np.concatenate([all_internal,np.zeros(num_lower)])
                for bd in neg_lower[0]:
                    print("{} Internal forward reactions forced on : {}\nBounds are ({},{})".format(self.Name,bd,np.minimum(-interior_lbs[bd],interior_ubs_min0[bd]),interior_ubs_min0[bd]))
                


            neg_upper = np.where(interior_ubs < 0) #need to add a constraint for the REVERSE reaction
            num_upper = len(neg_upper[0])
            if num_upper:
                ## Force on reverse
                fon += [(neg_upper[0][p] + len(interior_ubs),len(prob_mat) + p) for p in range(num_upper)]
                new_rows = np.concatenate([np.zeros((num_upper,num_v)),-np.eye(num_v)[neg_upper]],axis = 1)
                prob_mat = np.concatenate([std_form_mat,new_rows],axis = 0)
                all_internal = np.concatenate([all_internal,-np.minimum(-interior_ubs[neg_upper],interior_lbs_min0[neg_upper])])
                ## Force off forward
                foff += [(neg_upper[0][p],len(prob_mat) + p) for p in range(num_upper)]
                new_rows = np.concatenate([np.eye(num_v)[neg_upper],np.zeros((num_upper,num_v))],axis = 1)
                prob_mat = np.concatenate([std_form_mat,new_rows],axis = 0)
                all_internal = np.concatenate([all_internal,np.zeros(num_upper)])
                for bd in neg_upper[0]:
                    print("{} Internal reverse reactions forced on : {}\nBounds are ({},{})".format(self.Name,bd+len(interior_lbs),-np.minimum(interior_ubs[bd],interior_lbs_min0[bd]),interior_lbs_min0[bd]))

            self.ForcedOn = fon
            self.ForcedOff = foff

        else:
            self.ForcedOn = []
            self.ForcedOff = []

        std_form_mat = np.concatenate([prob_mat,np.eye(prob_mat.shape[0])],axis = 1)
        self.solver_constraint_matrix = prob_mat
        self.standard_form_constraint_matrix = std_form_mat
        self.internal_bounds = all_internal
        self.num_constr = std_form_mat.shape[0]
        self.total_vars = std_form_mat.shape[1]

        self.exchange_bounds_dt = np.concatenate([exterior_ubfuns_derivatives,exterior_lbfuns_derivative])




        self.deathrate = deathrate
        self.exchanged_metabolites = exchanged_metabolites

        self.current_basis_full = None
        self.current_basis = None

        self.essential_basis = None

        self.inter_flux = None
        self.slack_vals = None

        self.feasible = True


    def fba_gb(self,master_metabolite_con,secondobj = "total",report_activity = True,flobj = None):

        '''

        Perform FBA and minimize total flux, or use different secondary objective if given. Uses Gurobi.

        '''

        try:
            import gurobipy as gb
        except ImportError:
            print("Gurobipy import failed.")
            return None

        t1 = time.time()

        metabolite_con = master_metabolite_con[self.ExchangeOrder]

        solver_constraint_matrix = self.solver_constraint_matrix.copy()
        obje = self.objective

        # get the exchange bounds for the current metabolite environment
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])

        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        #Now we use Gurobi to solve min(x'obje) subject to Ax \leq b, x \geq 0
        #(with forward and reverse reaction fluxes separated)
        #


        if report_activity:
            try:
                flobj.write(self.Name + " [fba_gb] initializing LP\n")
            except:
                print(self.Name," [fba_gb] initializing LP")
        growth = gb.Model("growth")
        growth.setParam( 'OutputFlag', False )
        growth.setParam( 'Presolve', 0)


        allvars = growth.addMVar(solver_constraint_matrix.shape[1],lb = 0.0)
        growth.update()
        growth.setMObjective(None,obje,0,xc = allvars,sense = gb.GRB.MINIMIZE)
        growth.update()
        if report_activity:
            try:
                flobj.write(self.Name + " [fba_gb] Adding constraints\n")
            except:
                print(self.Name," [fba_gb] Adding constraints")

        growth.addMConstr(solver_constraint_matrix,allvars,"<=",bound_rhs)

        growth.update()

        if report_activity:
            try:
                flobj.write(self.Name + " [fba_gb] optimizing LP\n")
                flobj.write(self.Name + " [fba_gb] optimizing with " + str(len(growth.getConstrs())) + " constraints\n" )
            except:
                print(self.Name," [fba_gb] optimizing LP")
                print(self.Name," [fba_gb] optimizing with ",len(growth.getConstrs()) ," constraints" )
        growth.optimize()


        status = growth.status

        statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
        if status in statusdic.keys():
            if report_activity:
                try:
                    flobj.write(self.Name + " [fba_gb] LP Status: " +  statusdic[status] + '\n')
                except:
                    print(self.Name," [fba_gb] LP Status: ", statusdic[status])
        else:
            if report_activity:
                try:
                    flobj.write(self.Name + " [fba_gb] LP Status: Other\n")
                except:
                    print(self.Name," [fba_gb] LP Status: Other")

        if status == 2:

            fluxes = allvars.getAttr(gb.GRB.Attr.X)
            allconts = growth.getConstrs()
            slkvals = np.array([cn.getAttr(gb.GRB.Attr.Slack) for cn in allconts])
            all_vars = np.concatenate([fluxes,slkvals])

            basisinfo =  np.concatenate([np.array(allvars.getAttr(gb.GRB.Attr.VBasis)),np.array([cn.getAttr(gb.GRB.Attr.CBasis) for cn in allconts])])

            # print("Min all_vars0: {}".format(all_vars.min()))

            val = growth.getObjective().getValue()
            if report_activity:
                try:
                    flobj.write("{} [fba_gb] Initial Growth Rate: {}\n".format(self.Name,-val))
                except:
                    print("{} [fba_gb] Initial Growth Rate: {}".format(self.Name,-val))
            dosec = False

            if secondobj == "total":
                newobj = np.ones(self.num_fluxes)#,np.zeros(self.total_var - 2*self.num_fluxes)])
                dosec = True

            elif (type(secondobj) != str) and hasattr(secondobj, "__len__"):
                if len(secondobj) == self.num_fluxes:
                    newobj = secondobj#np.concatenate([secondobj,-secondobj])#,np.zeros(self.total_var - 2*self.num_fluxes)])
                    dosec = True

                elif len(secondobj) == int(0.5*self.num_fluxes):
                    newobj = np.concatenate([secondobj,-secondobj])#secondobj#np.concatenate([secondobj,np.zeros(self.total_var - 2*self.num_fluxes)])
                    dosec = True

                else:
                    print(self.Name," FBAWarning: Don't know what to make of given second objective. Give array of length corresponding to number of internal reactions (forward reverse separated: {}) or (not: {})".format(self.num_fluxes,int(0.5*self.num_fluxes)))

            if dosec:
                growth.addMConstr(np.array([obje]),allvars,"=",np.array([val]))
                growth.update()
                growth.setMObjective(None,newobj,0,xc = allvars,sense = gb.GRB.MINIMIZE)
                growth.update()

                # lbs = allvars.getAttr(gb.GRB.Attr.LB)
                # print("Lower bound = {}".format(min(lbs)))

                growth.optimize()
                growth.update()
                # print(growth.status)
                
                if growth.status == 2:
                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] Secondary Objective Value = {} \n".format(self.Name,growth.getObjective().getValue()))
                        except:
                            print("{} [fba_gb] Secondary Objective Value = {}".format(self.Name,growth.getObjective().getValue()))
                else: 

                    try:
                        flobj.write("{} [fba_gb] After secondary optimization status = {}\nIgnoring second optimization.\n".format(self.Name,statusdic[growth.status]))
                    except:
                        print("{} [fba_gb] After secondary optimization status = {}\nIgnoring second optimization.".format(self.Name,statusdic[growth.status]))

                    dosec = False
            

            #we need to get the non-zero variables and non-zero slacks. We don't need the entire bases - just the non-zeros. 
            if dosec:
                fluxes = allvars.getAttr(gb.GRB.Attr.X)

                allconts = growth.getConstrs()

                slkvals = np.array([cn.getAttr(gb.GRB.Attr.Slack) for cn in allconts])
                all_vars = np.concatenate([fluxes,slkvals])
                all_vars = all_vars[:-1]#don't include the added constraint

                basisinfo =  np.concatenate([np.array(allvars.getAttr(gb.GRB.Attr.VBasis)),np.array([cn.getAttr(gb.GRB.Attr.CBasis) for cn in allconts])])


            self.essential_basis = (all_vars>self.ezero).nonzero()[0]#



            
            if dosec:

                if basisinfo[-1]==0:
                    ##the secondary obj. slack is basic. Simply remove it.
                    thebasis = np.where(basisinfo[:-1] == 0)[0]
                else:

                    ##We need to add the secondary obj slack to the basis by pivoting so that we can then remove it (with no replacement)
                    # Really we just need to identify an index we can remove without changing either objective value.
                    augrow = np.concatenate([obje, np.zeros(self.solver_constraint_matrix.shape[0])])
                    augM = np.concatenate([self.standard_form_constraint_matrix,[augrow]],axis = 0)
                    augM = np.concatenate([augM, np.array([np.eye(augM.shape[0])[-1]]).T],axis = 1)
                    start_basis = np.where(basisinfo == 0)[0]

  
                    new_flxsb = np.linalg.solve(augM[:,start_basis],np.concatenate([bound_rhs,[val]]))
                    new_flxs = np.zeros(augM.shape[1])
                    new_flxs[start_basis] = new_flxsb

                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] Distance from all_vars: {}\n".format(self.Name,np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))
                        except:
                            print("{} [fba_gb] Distance from all_vars: {}".format(self.Name,np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))

                    new_growth_obj = np.dot(new_flxs[:len(obje)],obje)
                    new_total_flux = np.sum(new_flxs[:len(obje)])
                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] Start basis growth rate: {}\n Start basis total flux: {}".format(self.Name,-new_growth_obj,new_total_flux))
                        except:
                            print("{} [fba_gb] Start basis growth rate: {}\n Start basis total flux: {}".format(self.Name,-new_growth_obj,new_total_flux))

                    if np.min(new_flxs)<-10**-6:
                        print("{} [fba_gb] Start Basis - negative minimum flux: {} at index {}".format(self.Name,np.min(new_flxs),np.argmin(new_flxs)))

                    # neg_flx = np.where(new_flxs<-10**-6)[0]
                    # for nf in neg_flx:
                    #     print("Start Basis: Negative flux at {} = {}".format(nf,new_flxs[nf]))


                    Abarm1 = np.linalg.solve(augM[:,start_basis],augM[:,-1])
                    # csupp = np.concatenate([newobj,np.zeros(augM.shape[0])])
                    # cbarm1 = -np.dot(csupp[start_basis],Abarm1)
                    # print("cbarm1 = {}".format(cbarm1))
                    ### That should be 0...does it matter?
                    ### Choose i s.t. all_vars[start_basis[i]]=0 and Abarm1[i]>0
                    ## Compute the lambdas, choose the smallest. We get a little messed up by round off error
                    # leading to negative lambdas (from negative initial vars ~ -10**-7), so we'll take the abs of the vars to get the smallest lambda
                    # We have to choose over strictly positive Abars so fill negatives with -1 in resulting divide.
                    choicemaker = np.divide(abs(all_vars[start_basis]),Abarm1,-np.ones_like(Abarm1),where = Abarm1>10**-6)
                    choice = np.where(choicemaker == min(choicemaker[choicemaker>-0.5]))[0][0]

                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] Removed variable {} = {}.\n Value of lambda: {} \n Value of Abar: {}\n".format(self.Name,start_basis[choice],all_vars[start_basis][choice],choicemaker[choice],Abarm1[choice]))
                        except:
                            print("{} [fba_gb] Removed variable {} = {}.\n Value of lambda: {} \n Value of Abar: {}".format(self.Name,start_basis[choice],all_vars[start_basis][choice],choicemaker[choice],Abarm1[choice]))

                    thebasis = np.delete(start_basis,choice)

                    fllbasis = np.concatenate([thebasis,[augM.shape[1]-1]])

                    new_flxsb = np.linalg.solve(augM[:,fllbasis],np.concatenate([bound_rhs,[val]]))

                    new_flxs = np.zeros(augM.shape[1])
                    new_flxs[fllbasis] = new_flxsb

                    self.essential_basis = (new_flxs[:-1]>self.ezero).nonzero()[0]
                    
                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] Distance from all_vars: {}\n".format(self.Name,np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))
                        except:
                            print("{} [fba_gb] Distance from all_vars: {}".format(self.Name,np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))

                    fluxes = new_flxs[:len(fluxes)]
                    slkvals = new_flxs[:len(fluxes):-1]

                    

                    new_growth_obj = np.dot(new_flxs[:len(obje)],obje)
                    new_total_flux = np.sum(new_flxs[:len(obje)])

                    if report_activity:
                        try:
                            flobj.write("{} [fba_gb] New growth rate: {}\n New total flux: {}\n".format(self.Name,-new_growth_obj,new_total_flux))
                        except:
                            print("{} [fba_gb] New growth rate: {}\n New total flux: {}".format(self.Name,-new_growth_obj,new_total_flux))

                    if np.min(new_flxs)<-10**-6:
                        print("{} [fba_gb] New Basis negative minimum flux: {} at index {}".format(self.Name,np.min(new_flxs),np.argmin(new_flxs)))

                    # neg_flx = np.where(new_flxs<-10**-6)[0]
                    # for nf in neg_flx:
                    #     print("Negative flux at {} = {}".format(nf,new_flxs[nf]))


            else:
                thebasis = np.where(basisinfo == 0)[0]


            self.current_basis_full = thebasis

            
            self.inter_flux = fluxes
            self.slack_vals = slkvals


         



            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write(self.Name + " [fba_gb] Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print(self.Name," [fba_gb] Done in ",int(minuts)," minutes, ",sec," seconds.")


            return -val
        else:
            self.feasible = False
            return np.array(["failed to prep"])


    def fba_clp(self,master_metabolite_con,secondobj = "total",report_activity = True,flobj = None):

        '''

        Perform FBA and minimize total flux, or use different secondary objective if given. Uses CyLP (open source)

        '''
        from cylp.cy.CyClpSimplex import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray


        t1 = time.time()

        metabolite_con = master_metabolite_con[self.ExchangeOrder]

        solver_constraint_matrix = np.matrix(self.solver_constraint_matrix)
        obje = CyLPArray(self.objective)

        # get the exchange bounds for the current metabolite environment
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])

        bound_rhsarr = np.concatenate([exchg_bds,self.internal_bounds])
        bound_rhs = CyLPArray(bound_rhsarr)


        #Now we use clp to solve min(x'obje) subject to Ax = b, x \geq 0
        #the actual fluxes calculated are the first 2*num_v entries of x
        #(with forward and reverse reaction fluxes separated)
        #


        if report_activity:
            try:
                flobj.write(self.Name + " fba_clp: initializing LP\n")
            except:
                print(self.Name," fba_clp: initializing LP")
        growth = CyClpSimplex()

        x = growth.addVariable('x',solver_constraint_matrix.shape[1])



        if report_activity:
            try:
                flobj.write(self.Name + " fba_clp: Adding constraints\n")
            except:
                print(self.Name," fba_clp: Adding constraints")

        growth += solver_constraint_matrix * x <= bound_rhs

        growth.objective = obje * x

        if report_activity:
            try:
                flobj.write(self.Name + " fba_clp: optimizing LP\n")
                flobj.write(self.Name + " fba_clp: optimizing with " + str(solver_constraint_matrix.shape[0]) + " constraints\n" )
            except:
                print(self.Name," fba_clp: optimizing LP")
                print(self.Name," fba_clp: optimizing with ",solver_constraint_matrix.shape[0] ," constraints" )

        growth.variablesLower = np.zeros_like(growth.variablesLower)

        growth.primal()


        status = growth.getStatusString()


        if report_activity:
            try:
                flobj.write(self.Name + " fba_clp: LP Status: " +  status + '\n')
            except:
                print(self.Name," fba_clp: LP Status: ", status)


        if status == 'optimal':


            val = growth.objectiveValue
            dosec = False

            if secondobj == "total":
                newobj = CyLPArray(np.ones(self.num_fluxes))
                dosec = True

            elif (type(secondobj) != str) and hasattr(secondobj, "__len__"):
                if len(secondobj) == self.num_fluxes:
                    newobj = CyLPArray(secondobj)
                    dosec = True

                elif len(secondobj) == int(0.5*self.num_fluxes):
                    newobj = CyLPArray(np.concatenate([secondobj,-secondobj]))
                    dosec = True

                else:
                    print(self.Name," FBAWarning: Don't know what to make of given second objective")

            if dosec:
                growth += np.matrix([obje]) * x == val
                growth.objective = newobj * x
                growth.primal()



            x = np.array(growth.primalVariableSolution["x"])

            #Asking the solver for the slack values gave incorrect values...so that's odd.
            slkvals = bound_rhsarr - np.dot(self.solver_constraint_matrix,x)

            all_vars = np.concatenate([x,slkvals])

            self.essential_basis = (all_vars>self.ezero).nonzero()[0]

            basisinfo =  growth.varIsBasic

            if dosec:
                if basisinfo[-1]:
                    ##the secondary obj. slack is basic. Simply remove it.
                    thebasis = np.where(basisinfo[:-1])[0]
                else:




                    ##We need to add the secondary obj slack to the basis by pivoting so that we can then remove it (with no replacement)
                    # Really we just need to identify an index we can remove without changing either objective value.
                    augrow = np.concatenate([obje, np.zeros(self.solver_constraint_matrix.shape[0])])
                    augM = np.concatenate([self.standard_form_constraint_matrix,[augrow]],axis = 0)
                    augM = np.concatenate([augM, np.array([np.eye(augM.shape[0])[-1]]).T],axis = 1)
                    start_basis = np.where(basisinfo)[0]


                    # new_flxsb = np.linalg.solve(augM[:,start_basis],np.concatenate([bound_rhs,[val]]))
                    # new_flxs = np.zeros(augM.shape[1])
                    # new_flxs[start_basis] = new_flxsb

                    # print("Distance from all_vars: {}".format(np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))

                    # new_growth_obj = np.dot(new_flxs[:len(obje)],obje)
                    # new_total_flux = np.sum(new_flxs[:len(obje)])

                    # print("Start growth rate: {}\n New total flux: {}".format(-new_growth_obj,new_total_flux))

                    Abarm1 = np.linalg.solve(augM[:,start_basis],augM[:,-1])
                    # csupp = np.concatenate([newobj,np.zeros(augM.shape[0])])
                    # cbarm1 = -np.dot(csupp[start_basis],Abarm1)
                    # print("cbarm1 = {}".format(cbarm1))
                    ### That should be 0...does it matter?
                    ### Choose i s.t. all_vars[start_basis[i]]=0 and Abarm1[i]>0
                    ## We'll maximize Abarm1[i] to maximize the determinant of the resulting matrix.
                    #choicemaker = (1-all_vars[start_basis].round(6).astype(bool))*Abarm1
                    choicemaker = np.divide(all_vars[start_basis],Abarm1,-np.ones_like(Abarm1),where = Abarm1>0)
                    choice = np.where(choicemaker == min(choicemaker[choicemaker>0]))[0][0]

                    # print("Value of removed flux: {}".format(all_vars[start_basis][choice]))

                    if choicemaker[choice] > 0:
                        thebasis = np.delete(start_basis,choice)

                        fllbasis = np.concatenate([thebasis,[augM.shape[1]-1]])

                        new_flxsb = np.linalg.solve(augM[:,fllbasis],np.concatenate([bound_rhs,[val]]))

                        new_flxs = np.zeros(augM.shape[1])
                        new_flxs[fllbasis] = new_flxsb

                        self.essential_basis = (new_flxs[:-1]>self.ezero).nonzero()[0]

                        fluxes = new_flxs[:len(fluxes)]
                        slkvals = new_flxs[:len(fluxes):-1]

                        # print("Distance from all_vars: {}".format(np.linalg.norm(new_flxs - np.concatenate([fluxes,slkvals]))))

                        # new_growth_obj = np.dot(new_flxs[:len(obje)],obje)
                        # new_total_flux = np.sum(new_flxs[:len(obje)])

                        # print("New growth rate: {}\n New total flux: {}".format(-new_growth_obj,new_total_flux))

                        # neg_flx = np.where(new_flxs.round(6)<0)[0]
                        # for nf in neg_flx:
                        #     print("Negative flux at {} = {}".format(nf,new_flxs[nf]))

                    else:
                        print(choicemaker[choice])



            else:
                thebasis = np.where(basisinfo)[0]



            self.current_basis_full = thebasis


            self.inter_flux = x
            self.slack_vals = slkvals

            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write(self.Name + " fba_clp: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print(self.Name," fba_clp: Done in ",int(minuts)," minutes, ",sec," seconds.")



            return -val
        else:
            self.feasible = False
            return np.array(["failed to prep"])


    def findWave(self,master_metabolite_con,master_metabolite_con_dt,details = False, flobj = None):

        all_current_vars = np.concatenate([self.inter_flux,self.slack_vals])

        metabolite_con = master_metabolite_con[self.ExchangeOrder]
        metabolite_con_dt = master_metabolite_con_dt[self.ExchangeOrder]

        exchg_bds_dt = np.array([bd(metabolite_con,metabolite_con_dt) for bd in self.exchange_bounds_dt])
        bound_rhs_dt = np.concatenate([exchg_bds_dt,np.zeros(len(self.internal_bounds))]).round(8)

        basisinds = self.current_basis_full.copy()

        # svd = np.linalg.svd(self.standard_form_constraint_matrix[:,basisinds],compute_uv = False)
        # print("Minimum singular value: {}".format(min(svd)))

        essential_indx = np.array([i for i in range(len(basisinds)) if (basisinds[i] in self.essential_basis)])#location of essentials within beta
        
        Abeta = self.standard_form_constraint_matrix[:,basisinds]
        Vbeta = np.linalg.solve(Abeta,bound_rhs_dt)

        # ######


        #If we're lucky we already have a basic feasible. For now, that's good enough. I'm not trying to optimize something like "most interior" or whatever.
        # print("Min border dv/dt = {}".format(min([Vbeta[i] for i in range(len(Vbeta)) if (i not in essential_indx)])))

        UpDateFlag = True

        if all([Vbeta[i]>-self.ezero for i in range(len(Vbeta)) if (i not in essential_indx)]):
        # if all([Vbeta[i]>-10**-6 for i in range(len(Vbeta)) if (i not in essential_indx)]):
            UpDateFlag = False
            if details:
                try:
                    flobj.write("{}.findWaves: No Pivot Needed \n".format(self.Name))
                except:
                    print("{}.findWaves: No Pivot Needed".format(self.Name))
            basisinds.sort()
            self.current_basis_full = basisinds
            if details:
                try:
                    flobj.write("{}.findWaves: Solving min-max\n".format(self.Name))
                except:
                    print("{}.findWaves: Solving min-max".format(self.Name))

            UpDateFlag = self.solve_minmax(basisinds,bound_rhs_dt,all_current_vars,essential_indx,details,flobj)

            return None

        #Otherwise, we can use the simplex algorithm with the "phase-one" problem.

        if details:
            try:
                flobj.write("{}.findWaves: Solving phase-1\n".format(self.Name))
            except:
                print("{}.findWaves: Solving phase-1".format(self.Name))

        basisinds,objval = self.solve_phaseone(Abeta,Vbeta,essential_indx,basisinds,details,bound_rhs_dt,flobj)


        #### Note that if the objective value of the phase 1 problem is > 0 then the problem is infeasible.
        if objval == 0:
            basisinds.sort()
            essential_indx = np.array([i for i in range(len(basisinds)) if (basisinds[i] in self.essential_basis)])

            #Finally we can choose from bases that will allow forward solving by trying to maximize the linear estimate of the forward solve interval
            #by maximizing the the minimum of the linear estimates for each variable 
            #(linear estimate is possible b/c our solution gives dv/dt and interval ends when v = 0 for any v)
            if details:
                try:
                    flobj.write("{}.findWaves: Solving min-max\n".format(self.Name))
                except:
                    print("{}.findWaves: Solving min-max".format(self.Name))
            _ = self.solve_minmax(basisinds,bound_rhs_dt,all_current_vars,essential_indx,details,flobj)



        else:
            self.feasible = False
            UpDateFlag = False
            print("{}.findWaves: No feasible basis for forward simulation. Objective value stalled at {}".format(self.Name,objval))


        return UpDateFlag

    def solve_phaseone(self,Abeta,Vbeta,essential_indx,basisinds,details,bound_rhs_dt,flobj):

        Abarnp1 = np.array([np.dot(-Abeta,np.ones(Abeta.shape[0]))]).T
        Aplus = np.concatenate([self.standard_form_constraint_matrix,Abarnp1],axis = 1)
        #get most negative entry that isn't in the essential basis and switch it with n+1
        vdummy = Vbeta.copy()
        vdummy[essential_indx] = max(1,max(Vbeta))
        #find min
        min_nonessential = np.argmin(vdummy)
        basisinds[min_nonessential] = self.standard_form_constraint_matrix.shape[1]

        alldone = False

        pivcnt = 0
        if details:
            objval = np.linalg.solve(Aplus[:,basisinds],bound_rhs_dt)[min_nonessential]
            try:
                flobj.write("{}.solve_phase1: Initial Objective Value of Phase-0 Problem: {}".format(self.Name,objval))
            except:
                print("{}.solve_phase1: Initial Objective Value of Phase-0 Problem: {}".format(self.Name,objval))
        changing = np.ones(10)
        btol = 10**-8
        while (((not alldone) and (pivcnt < 1000)) and (np.mean(changing)>btol)):
        #returns in as indexed in A, out as indexed in beta
            pivin,pivout,alldone,ch = pivot(np.eye(Aplus.shape[1])[-1],Aplus,basisinds,bound_rhs_dt,min_nonessential,essential_indx)
            pivout_ind = basisinds[pivout]

            if pivout_ind in self.essential_basis and pivout_ind != pivin:
                print("{}.solve_phase1: What's wrong with index {}\nThats {} in beta\nIs that in the list? {}".format(self.Name,pivout_ind,pivout,pivout in essential_indx))
                sys.exit()
            basisinds[pivout] = pivin
            changing[1:] = changing[:-1]
            changing[0] = abs(ch)
            if ch == 0:
                changing = np.zeros_like(changing)
            if not alldone:
                objval = np.linalg.solve(Aplus[:,basisinds],bound_rhs_dt)[min_nonessential]
                if details:
                    try:
                        flobj.write("{}.solve_phase1: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {} \n".format(self.Name,pivcnt+1,pivin,pivout_ind,objval))
                    except:
                        print("{}.solve_phase1: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {}".format(self.Name,pivcnt+1,pivin,pivout_ind,objval))
            else:
                objval = 0
                if details:
                    try:
                        flobj.write("{}.solve_phase1: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {} \n".format(self.Name,pivcnt+1,pivin,pivout_ind,0))
                    except:
                        print("{}.solve_phase1: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {}".format(self.Name,pivcnt+1,pivin,pivout_ind,0))
        
        
            pivcnt += 1
        return basisinds,objval

    def solve_minmax(self,basisinds,bound_rhs_dt,all_current_vars,essential_indx,details,flobj,constrained = False):

        keeptrying = True
        numof = 0

        wbeta = np.linalg.solve(self.standard_form_constraint_matrix[:,basisinds],bound_rhs_dt)
        w = np.zeros(self.standard_form_constraint_matrix.shape[1])
        w[basisinds] = wbeta
        thevs = -np.divide(np.ones_like(all_current_vars),all_current_vars,out = np.zeros_like(all_current_vars), where = np.abs(all_current_vars)>10**-8)

        oneovertimeto = thevs*w


        # print("========== {} ==========".format("How Bad (start)?"))
        # print(max(oneovertimeto))
        # print("====================")

        maxtimes = [max(oneovertimeto)]


        stopat = 10**-10

        pivcnt = 0

        while keeptrying:
            UpDateFlag = True
            if constrained:
                pivin,pivout,alldone,_ = minmaxpivot_constrained(w,thevs,self.standard_form_constraint_matrix,basisinds,bound_rhs_dt,essential_indx)
            else:
                pivin,pivout,alldone,_ = minmaxpivot(w,thevs,self.standard_form_constraint_matrix,basisinds,bound_rhs_dt,essential_indx)
            pivout_ind = basisinds[pivout]
            if pivout_ind in self.essential_basis and pivout_ind != pivin:
                print("{}.solve_minmax: What's wrong with index {}\nThats {} in beta\nIs that in the list? {}".format(self.Name,pivout_ind,pivout,pivout in essential_indx))
                sys.exit()
            basisinds[pivout] = pivin
            basisinds.sort()
            essential_indx = np.array([i for i in range(len(basisinds)) if (basisinds[i] in self.essential_basis)])


            Abeta = self.standard_form_constraint_matrix[:,basisinds]
            wbeta = np.linalg.solve(Abeta,bound_rhs_dt)
            w = np.zeros(self.standard_form_constraint_matrix.shape[1])
            w[basisinds] = wbeta

            oneovertimeto = w*thevs#
            numof+=1

            # print("========== {} - {} ==========".format("How Bad?",numof))
            # print(max(oneovertimeto))
            # print("====================")
            maxtimes += [max(oneovertimeto)]

            pivcnt += 1
            if details:
                try:
                    flobj.write("{}.solve_minmax: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {} \n".format(self.Name,pivcnt,pivin,pivout_ind,maxtimes[-1]))
                except:
                    print("{}.solve_minmax: Pivot number {}, Pivot In: {}, Pivot Out: {}, Objective: {}".format(self.Name,pivcnt,pivin,pivout_ind,maxtimes[-1]))

            keeptrying = not alldone
            #stop if we've done so well - this will also get us out of any loop.
            if maxtimes[-1] <= stopat:
                keeptrying = False
            elif numof > 100:
                keeptrying = False
            elif numof > 5:
                #stop if we're stuck at the same number
                if np.std(maxtimes[-5:]) < 10**-5:
                    keeptrying = False
            #also stop if we're stuck in a loop
            elif numof > 2:
                if maxtimes[-1] in maxtimes[:-1]:
                    #get the loop:
                    loopat = np.where(np.array(maxtimes) == maxtimes[-1])[0]
                    if maxtimes[-1] == min(maxtimes[loopat[0]:loopat[1]]):
                        keeptrying = False
                    else:
                        stopat = min(maxtimes[loopat[0]:loopat[1]])


        if numof  == 1:
            UpDateFlag = False

        self.current_basis_full = basisinds
        self.current_basis = getReduced(basisinds,self.num_fluxes,self.standard_form_constraint_matrix)


        return UpDateFlag
            
    def compute_internal_flux(self,master_metabolite_con):

        '''
        Compute current fluxes (not including slacks) from current basis & metabolite concentration - uses reduced basis.
        '''
        metabolite_con = master_metabolite_con[self.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        Q,R,beta = self.current_basis

        if len(beta[0]):
            fl_beta = la.solve_triangular(R,np.dot(Q.T,bound_rhs[beta[0]]))
        else:
            fl_beta = np.array([])

        all_vars = np.zeros(self.num_fluxes)#np.zeros(self.total_var)
        all_vars[beta[1]] = fl_beta

        self.inter_flux = all_vars#compute_if(bound_rhs.astype(np.float64),self.current_basis,self.num_fluxes)

    def compute_slacks(self,master_metabolite_con):

        '''Compute the values of the slack variables'''

        metabolite_con = master_metabolite_con[self.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        # Q,R,beta = self.current_basis

        # all_slacks = np.zeros(self.num_constr)

        # basic_slacks = np.array([i for i in range(self.num_constr) if i not in beta[0]])

        # basic_slk_bds = bound_rhs[basic_slacks]
        # basic_slk_vals = basic_slk_bds - np.dot()

        all_slks = bound_rhs - np.dot(self.solver_constraint_matrix,self.inter_flux)

        self.slack_vals = all_slks

@jit(nopython=True)
def compute_if(bound_rhs,current_basis,num_fluxes):
    '''
    Compute current fluxes (not including slacks) from current basis & metabolite concentration - uses reduced basis.
    '''
    # metabolite_con = master_metabolite_con[ExchangeOrder]
    # exchg_bds = np.array([bd(metabolite_con) for bd in exchange_bounds])
    # bound_rhs = np.concatenate([exchg_bds,internal_bounds])

    Q,R,beta = current_basis

    if len(beta[0]):
        fl_beta = np.linalg.solve(R,np.dot(Q.T,bound_rhs[beta[0]]))
    else:
        fl_beta = np.empty(0,np.float64)

    all_vars = np.zeros(num_fluxes)#np.zeros(self.total_var)
    all_vars[beta[1]] = fl_beta

    return all_vars

def getReduced(basisinds,num_fluxes,A):
    rowsToDelete = np.array([i-num_fluxes for i in basisinds if i>=num_fluxes])
    rowsToInclude = np.array([i for i in range(A.shape[0]) if i not in rowsToDelete])
    colsToInclude = np.array([i for i in basisinds if i<num_fluxes])
    reducedbeta = (rowsToInclude,colsToInclude)
    if len(rowsToInclude):
        reducedAbeta = A[rowsToInclude][:,colsToInclude]
        Q,R = np.linalg.qr(reducedAbeta)
    else:
        Q = np.array([[]])
        R = np.array([[]])
        reducedbeta = (list(rowsToInclude),list(colsToInclude))
    return (Q,R,reducedbeta)

def remove_licol(fullmat,indexes,essentials,fluxes):
    mat = fullmat[:,indexes]
    if mat.shape[0] == mat.shape[1]:
        return indexes,None
    slv = la.null_space(mat).T[0]
    non_essentials = np.array([x for x in range(len(indexes)) if indexes[x] not in essentials])
    #and remove any column with a non-zero, as long as it isn't ''essential"
    nonzero = np.where(slv.round(7).astype(bool)*np.array([(x not in essentials) for x in indexes]))[0]#[-1]

    # all_errs = np.array([np.linalg.norm(fluxes[j]*np.delete(slv,j)/slv[j]) if slv[j].round(10) else 100 for j in range(len(slv))])
    all_errs = np.array([np.linalg.norm(np.delete(slv,j)/slv[j]) if slv[j].round(10) else 100 for j in range(len(slv))])

    if len(nonzero):
        # remove where it will have the smallest error. Error vector is er = v * np.linalg.solve(A,y) where
        # v is the flux value at the index removed, y is the 
        # column removed and A is the remaining matrix. 
        rmvit = nonzero[np.argmin(all_errs[nonzero])]
        # rmvit = nonzero[np.argmax(slv[nonzero])]
        return np.delete(indexes,rmvit),rmvit
    else:
        print("Failed to find initial basis after secondary objective.")
        return None,None


def pivot_nojit(c,A,beta,b,preferred,muststay = None):
    if muststay == None:
        muststay = []
    #get eta
    eta = np.array([i for i in range(A.shape[1]) if (i not in beta)])
    #compute A^-1_beta * A
    Abar = np.linalg.solve(A[:,beta],A)
    #compute reduced costs
    cbar = c - np.dot(c[beta].T,Abar)
    #compute primal basic solution
    xbeta = np.linalg.solve(A[:,beta],b)
    #get the possible pivot ins
    negcbareta = np.array([i for i in eta if cbar[i] < 0])
    ###Can't choose a column with only negative values or only positive on essentials. Remove those from negchareta
    negcbareta = np.array([i for i in negcbareta if np.any(Abar[np.array([j for j in range(Abar.shape[0]) if j not in muststay]),i]>10**-6)])
    #get the pivot out values
    Abarnegeta = Abar[:,negcbareta]
    alllambdas = np.divide(xbeta,Abarnegeta.T,np.zeros_like(Abarnegeta.T),where= Abarnegeta.T > 10**-6)
    maxlam = np.max(alllambdas) + 1    
    alllambdas[np.where(Abarnegeta.T <= 10**-6)] = maxlam#np.max(alllambdas) + 1
    ###Don't want to pivot some of them out. Also we are ok with those ones going negative!
    alllambdas[:,muststay] = maxlam#np.max(alllambdas) #add to COLUMNS because we've taken a transpose.
    # print(np.max(alllambdas))
    istars = np.argmin(alllambdas,axis = 1)
    ### Stopping condition is to pivot out the preferred variable. This can happen when lambda[i,preferred] is minimal. Sometimes, it won't be in istars b/c argmin gives the first one.
    checkfordone = np.array([np.isclose(alllambdas[i,preferred],alllambdas[i,istars[i]],atol = 10**-5) for i in range(alllambdas.shape[0])]) #preferred is index of preferred in beta, not in A
    if np.any(checkfordone):
        in0 = np.where(checkfordone)[0][0]
        # print("Rank check is {}\n Chosen with {}".format(Abarnegeta[istars[in0],in0],alllambdas[in0,istars[in0]]))
        #return (in,out)
        pivotin = negcbareta[in0]
        return pivotin,preferred,True
    #if we can't pivot out the one we really want to choose the one that has the largest effect I guess.
    all_lambdas = np.array([alllambdas[i,istars[i]] for i in range(len(istars))])
    changes = all_lambdas*cbar[negcbareta]
    in0 = np.argmin(changes)#index of the entering one in Abarnegeta.
    out = istars[in0]#index of leaving (in beta, not A)
    pivotin = negcbareta[in0]
    # print("Rank check is {}\n Chosen with {}".format(Abarnegeta[istars[in0],in0],alllambdas[in0,istars[in0]]))
    return pivotin,out,False

@jit(nopython=True)
def minmaxpivot(x,v,A,beta,b,muststay):
    ''''
    pivot to reduce max(v'x) - actually just going to reduce the element that is currently max. Could actually
    increase the overall max...is there a better way?
    '''

    if max(v) < 10**-8:
        return beta[0],0,True,0
    
    #get eta
    eta = np.array([i for i in range(A.shape[1]) if (i not in beta)])
    #compute A^-1_beta * A
    Abar = np.linalg.solve(A[:,beta],A)
    #compute primal basic solution
    xbeta = x[beta]
    ##multiply by v
    maxof = v*x
    maxind = maxof.argmax()
    c = np.zeros_like(v)
    c[maxind] = v[maxind]
    #compute reduced costs
    cbar = c - np.dot(c[beta].T,Abar)

    #get the possible pivot ins
    negcbareta = np.array([i for i in eta if cbar[i] < 0])
    ###Can't choose a column to pivot in with only negative values or only positive on essentials. Remove those from negchareta
    ## This is because our pivot out i^* must have Abar[eta_j,i^*] > 0 and i^*  cannot be essential.
   
    canleave = np.array([i for i in range(A.shape[0]) if i not in muststay])
    negcbareta = np.array([i for i in negcbareta if max([a for a in Abar[canleave,i]]) > 10**-6])

    #get the columns of Abar that correspond to ok pivot in based on cbar_eta.
    Abarnegeta = Abar[:,negcbareta]

    ### The possible lambdas x_{out}/Abar[out,in]
    alllambdas = np.divide(xbeta,Abarnegeta.T,np.zeros_like(Abarnegeta.T))#, Abarnegeta.T > 10**-6)
    # maxlam = np.max(alllambdas)
    ###Don't want to pivot some of them out. Also we are ok with those ones going negative! 
    # 
    #    
    istars = np.array([conditionargmin(alllambdas[i],[x>10**-6 for x in Abarnegeta.T[i]],muststay) for i in range(len(alllambdas))])

    #if istars[j] == alllambdas.shape[1], then we couldn't find an argmin and that column should not have been in negcbareta
    for k in range(len(istars)):
        if k == alllambdas.shape[1]:
            negcbareta = np.delete(negcbareta,k)
            istars = np.delete(istars,k)
            alllambdas = alllambdas[np.delete(np.arange(len(alllambdas)),k)]

    if len(negcbareta) == 0:
        return beta[0],0,True,0 

    all_lambdas = np.array([alllambdas[i,istars[i]] for i in range(len(istars))])
    changes = all_lambdas*cbar[negcbareta]
    # print(min(changes))
    if min(changes) > -10**-8:
        return beta[0],0,True,0


    in0 = np.argmin(changes)#index of the entering one in Abarnegeta.
    out = istars[in0]#index of leaving (in beta, not A)
    pivotin = negcbareta[in0]


    # print("Rank check is ",Abar[out,pivotin],"\n Should change obj. by ",changes[in0])
    return pivotin,out,False,changes[in0]

@jit(nopython=True)
def pivot(c,A,beta,b,preferred,muststay):

    #get eta
    eta = np.array([i for i in range(A.shape[1]) if (i not in beta)])
    #compute A^-1_beta * A
    Abar = np.linalg.solve(A[:,beta],A)
    #compute reduced costs
    cbar = c - np.dot(c[beta].T,Abar)
    #compute primal basic solution
    xbeta = np.linalg.solve(A[:,beta],b)
    #get the possible pivot ins
    negcbareta = np.array([i for i in eta if cbar[i] < 0])
    ###Can't choose a column to pivot in with only negative values or only positive on essentials. Remove those from negchareta
    ## This is because our pivot out i^* must have Abar[eta_j,i^*] > 0 and i^*  cannot be essential.
   
    canleave = np.array([i for i in range(A.shape[0]) if i not in muststay])
    negcbareta = np.array([i for i in negcbareta if max([a for a in Abar[canleave,i]]) > 10**-6])


    
    #get the columns of Abar that correspond to ok pivot in based on cbar_eta.
    Abarnegeta = Abar[:,negcbareta]

    ### The possible lambdas x_{out}/Abar[out,in]
    alllambdas = np.divide(xbeta,Abarnegeta.T,np.zeros_like(Abarnegeta.T))#, Abarnegeta.T > 10**-6)
    # maxlam = np.max(alllambdas)
    ###Don't want to pivot some of them out. Also we are ok with those ones going negative! 
    # 
    #    
    istars = np.array([conditionargmin(alllambdas[i],[x>10**-6 for x in Abarnegeta.T[i]], muststay) for i in range(len(alllambdas))])

    #if istars[j] == alllambdas.shape[1], then we couldn't find an argmin and that column should not have been in negcbareta
    for k in range(len(istars)):
        if k == alllambdas.shape[1]:
            negcbareta = np.delete(negcbareta,k)
            istars = np.delete(istars,k)
            alllambdas = alllambdas[np.delete(np.arange(len(alllambdas)),k)]
    
    if len(istars) == 0:
        return beta[0],0,False,0 


        
    ### Stopping condition is to pivot out the preferred variable. This can happen when lambda[i,preferred] is minimal. Sometimes, it won't be in istars b/c argmin gives the first one.
    checkfordone = np.array([abs(alllambdas[i,preferred] - alllambdas[i,istars[i]]) < 10**-5 for i in range(alllambdas.shape[0])]) 
    #preferred is index of preferred in beta, not in A

    if np.any(checkfordone):
        in0 = np.where(checkfordone)[0][0]
        pivotin = negcbareta[in0]
        # print("Rank check is ",Abar[preferred,pivotin],"\n Chosen with ",alllambdas[in0,preferred])
        return pivotin,preferred,True,alllambdas[in0,preferred]*cbar[pivotin]

    #if we can't pivot out the one we really want to choose the one that has the largest effect I guess.
    all_lambdas = np.array([alllambdas[i,istars[i]] for i in range(len(istars))])
    changes = all_lambdas*cbar[negcbareta]
    in0 = np.argmin(changes)#index of the entering one in Abarnegeta.
    out = istars[in0]#index of leaving (in beta, not A)
    pivotin = negcbareta[in0]


    # print("Rank check is ",Abar[out,pivotin],"\n Should change obj. by ",changes[in0])
    return pivotin,out,False,changes[in0]

@jit(nopython=True)
def conditionargmin(arr,condition_arr,exclude):
    ok_inds = np.where(np.array(condition_arr))[0]
    ok_inds = np.array([i for i in ok_inds if i not in exclude])
    if len(ok_inds):
        amin = ok_inds[0]
        for j in ok_inds:
            if j not in exclude:
                if arr[j] < arr[amin]:
                    amin = j
        return amin
    else:
        print("No qualifying min found.")
        return len(arr)



def compute_objval(c,A,beta,b):
    xbeta = np.linalg.solve(A[:,beta],b)
    x = np.zeros_like(c)
    x[beta] = xbeta
    return np.dot(c,x)

    
@jit(nopython=True)
def minmaxpivot_constrained(w,v,A,beta,b,muststay):
    ''''
    pivot to reduce max(v*x) - actually just going to reduce the element that is currently max. Constrain the other elements of v*x
    by the current max. This ends up looking like a single pivot of an LP solve but we have to add particular constraints each time.
    '''

    #compute primal basic solution
    wbeta = w[beta]

    ##multiply by v
    s = v*w
    maxind = s.argmax()
    smax = np.max(s)

    ### Now add the new constraints onto A,b

    Apl = np.append(A,np.zeros(A.shape),axis = 1)
    eyeye = np.concatenate((np.eye(A.shape[1]),np.eye(A.shape[1])),axis = 1)
    Apl = np.append(Apl,eyeye,axis = 0)

    bpl = np.append(b,np.array([smax]*A.shape[1]))

    betapl = np.append(beta,np.arange(A.shape[1],Apl.shape[1]))
    wpl = np.append(w,smax - w)


    #get eta
    eta = np.array([i for i in range(A.shape[1]) if (i not in beta)]) #equivalent to np.array([i for i in range(Apl.shape[1]) if (i not in betapl)])
    #compute A^-1_betapl * A
    Abar = np.linalg.solve(Apl[:,betapl],Apl)

    c = np.zeros(Apl.shape[1])
    c[maxind] = 1#v[maxind]
    #compute reduced costs
    cbar = c - np.dot(c[betapl].T,Abar)

    #get the possible pivot ins
    negcbareta = np.array([i for i in eta if cbar[i] < 0])
    if len(negcbareta) == 0:
        return beta[0],0,True,0 
    ###Can't choose a column to pivot in with only negative values or only positive on essentials (including new constraints). Remove those from negchareta
    ## This is because our pivot out i^* must have Abar[eta_j,i^*] > 0 and i^*  cannot be essential.
    muststaypl = np.append(muststay,np.arange(A.shape[1],Apl.shape[1]))
   
    canleave = np.array([i for i in range(A.shape[0]) if i not in muststay])#equivalent to np.array([i for i in range(Apl.shape[0]) if i not in muststaypl])
    negcbareta = np.array([i for i in negcbareta if max([a for a in Abar[canleave,i]]) > 10**-6])

    #get the columns of Abar that correspond to ok pivot in based on cbar_eta.
    Abarnegeta = Abar[:,negcbareta]

    ### The possible lambdas x_{out}/Abar[out,in]
    alllambdas = np.divide(wpl[betapl],Abarnegeta.T,np.zeros_like(Abarnegeta.T))#, Abarnegeta.T > 10**-6)
    # maxlam = np.max(alllambdas)
    ###Don't want to pivot some of them out. Also we are ok with those ones going negative! 
    # 
    #    
    istars = np.array([conditionargmin(alllambdas[i],[x>10**-6 for x in Abarnegeta.T[i]],muststaypl) for i in range(len(alllambdas))])

    #if istars[j] == alllambdas.shape[1], then we couldn't find an argmin and that column should not have been in negcbareta
    for k in range(len(istars)):
        if k == alllambdas.shape[1]:
            negcbareta = np.delete(negcbareta,k)
            istars = np.delete(istars,k)
            alllambdas = alllambdas[np.delete(np.arange(len(alllambdas)),k)]

    if len(negcbareta) == 0:
        return beta[0],0,True,0 

    all_lambdas = np.array([alllambdas[i,istars[i]] for i in range(len(istars))])
    changes = all_lambdas*cbar[negcbareta]
    print(min(changes))
    if min(changes) > -10**-8:
        return beta[0],0,True,0


    in0 = np.argmin(changes)#index of the entering one in Abarnegeta.
    out = istars[in0]#index of leaving (in beta, not A)
    pivotin = negcbareta[in0]


    # print("Rank check is ",Abar[out,pivotin],"\n Should change obj. by ",changes[in0])
    return pivotin,out,False,changes[in0]