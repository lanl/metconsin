from surfmod import *

def get_waves(models,mets,mets0):
    waves = {}
    m0 = [mets0[ky] for ky in mets]
    wi = {}
    for md in models.keys():
        wi[md] = models[md].prep_indv_model(m0)
    yd0 = sum([np.dot(np.concatenate([models[md].Gamma1,-models[md].Gamma1],axis = 1),wi[md]) for md in wi.keys()])
    for md in models.keys():
        print(md)
        bddts = np.concatenate([models[md].uptakes*yd0,np.zeros(len(models[md].MatrixA) - len(yd0))])
        Q,R,indx,_ = find_waves(m0,wi[md],bddts,models[md],report_activity = True, solver = "coin")
        waves[md] = (Q,R,indx)
    return waves

def find_waves(y,v,bddts,surfmodel,headsup = [], model = None,report_activity = True, solver = 'coin',flobj = None):
    '''
    bddts must be 0 (for internal bound) or kappa_j ydot_j

    MatA is
    [-Gamma1 ]
    [Gamma1]
    [   I   ]
    [  -I   ]
    where Gamma1 is GammaStar and Gamma2 is GammaDagger

    ******** If 'coin' is used (and this will be changed to all solvers if possible)
    Positive and negative fluxes are split (so that total flux can be minimized initially) so A becomes
     [A,-A]
     [I, 0]
     [0, I]

     Gamma2 becomes [Gamma2,-Gamma2], etc.

    statbds needs to be (negative) lower exchange then upper internal then (negative) lower internal


    Returns a basis matrix B and ind so vdot is solution to solve(B,bddts[ind])
    '''



    t = time.time()
    precision = 8
    minprecision = 0
    MatA = surfmodel.MatrixA
    if solver == 'coin':
        MatA = np.concatenate([MatA,-MatA],axis = 1)
        MatA = np.concatenate([MatA,-np.eye(MatA.shape[1])],axis = 0)
    statbds = surfmodel.statbds
    kappas = surfmodel.uptakes
    Gamma1 = surfmodel.Gamma1
    Gamma2 = surfmodel.Gamma2
    if solver == 'coin':
        Gamma2 = np.concatenate([Gamma2,-Gamma2],axis = 1)
    obje = surfmodel.objective
    if solver == 'coin':
        obje = np.concatenate([obje,-obje])
    death = surfmodel.deathrate
    ###Determine which constraints we are at
    bds = np.concatenate([y*kappas,statbds])
    if solver == 'coin':
        bds = np.concatenate([bds,np.zeros(MatA.shape[1])])
        bddts = np.concatenate([bddts,np.zeros(MatA.shape[1])])


    dists = np.dot(MatA,v) - bds
    if len(headsup):
        dists[headsup] = 0.0
    atthem = ([],)#

    LP_not_ready = True

    if report_activity:
        try:
            flobj.write("find_waves: Initializing LP\n")
        except:
            print("find_waves: Initializing LP")

    if solver == 'gb':

        if model == None:

            findbasis = gb.Model("findbasis")
            findbasis.setParam( 'OutputFlag', False )

            if report_activity:
                try:
                    flobj.write("find_waves: Adding Variables\n")
                except:
                    print("find_waves: Adding Variables")
                taddk = time.time()
            vdots = [findbasis.addVar(lb = - gb.GRB.INFINITY,ub = gb.GRB.INFINITY, name = "s" + str(i)) for i in range(MatA.shape[1])]
            findbasis.update()
            objv = gb.quicksum([a[0]*a[1] for a in zip(obje,vdots)])
            findbasis.setObjective(objv,gb.GRB.MAXIMIZE)
            findbasis.addConstrs((gb.quicksum([Gamma2[i][l]*vdots[l] for l in range(len(vdots))]) == 0 for i in range(len(Gamma2))), name = 'Kernal')
            if report_activity:
                try:
                    flobj.write("find_waves: Added Kernal constraints in " + str(time.time() - taddk) + " seconds.\n")
                except:
                    print("find_waves: Added Kernal constraints in ",time.time() - taddk, " seconds.")
            inthere = []

        else:
            if report_activity:
                try:
                    flobj.write("find_waves: Removing Constraints\n")
                except:
                    print("find_waves: Removing Constraints")
            findbasis = model
            findbasis.reset(1)
            vdots = findbasis.getVars()
            inthere = [con.ConstrName for con in [findbasis.getConstrByName('LE[' + str(i) + ']') for i in range(len(bddts))] if con != None]

    elif solver == 'cp':
        if model == None:

            findbasis = cp.Cplex()
            if report_activity:
                try:
                    flobj.write("find_waves: Adding Variables\n")
                except:
                    print("find_waves: Adding Variables")
                taddk = time.time()
            vdots = ["s" + str(i) for i in range(MatA.shape[1])]
            lbs = [-cp.infinity]*len(vdots)
            ups = [cp.infinity]*len(vdots)
            findbasis.variables.add(obj = obje,lb = lbs,ub = ups,names = vdots)
            findbasis.objective.set_sense(findbasis.objective.sense.maximize)

            findbasis.set_results_stream(None)
            findbasis.set_warning_stream(None)

            constrmat = [[vdots,list(g2)] for g2 in Gamma2.astype(float)]
            findbasis.linear_constraints.add(lin_expr = constrmat, senses = ['E']*len(Gamma2), rhs = [0.0]*len(Gamma2), names = ['Kernal['+str(i)+']' for i in range(len(Gamma2))])

            inthere = []

        else:
            if report_activity:
                try:
                    flobj.write("find_waves: Removing Constraints\n")
                except:
                    print("find_waves: Removing Constraints")
            findbasis = model
            vdots = findbasis.variables.get_names()
            inthere = [nm for nm in findbasis.linear_constraints.get_names() if 'LE' in nm]

    elif solver == 'coin':
        # if model == None:
        findbasis = coin.cy.CyClpSimplex()
        if report_activity:
            try:
                flobj.write("find_waves: Adding Variables\n")
            except:
                print("find_waves: Adding Variables")
            taddk = time.time()

        vdots = findbasis.addVariable('vdots',MatA.shape[1])
        findbasis.optimizationDirection = "max"
        findbasis.addConstraint(np.matrix(Gamma2)*vdots == 0, name = "Kernal")
        cobj = coin.py.modeling.CyLPArray(obje)
        findbasis.objective = cobj * vdots

        if report_activity:
            try:
                flobj.write("find_waves: Added Kernal constraints in " + str(time.time() - taddk) + " seconds.\n")
            except:
                print("find_waves: Added Kernal constraints in ",time.time() - taddk, " seconds.")


    else:
        print("Please select solver Coin-CBC: 'coin', Gurobi: 'gb' or CPlex: 'cp'")
        return None

    while (LP_not_ready) and (precision >= minprecision):

        atthem = np.where(dists.round(precision) >= 0)


        the_rows_of_A = MatA[atthem]


        A_res_and_G2 = np.concatenate([the_rows_of_A,Gamma2])



        if np.linalg.matrix_rank(A_res_and_G2) < MatA.shape[1] and precision > minprecision:
            precision = precision - 1
            print("find_wave: lowering precision: ",precision)

        elif np.linalg.matrix_rank(A_res_and_G2) < MatA.shape[1]:
            print('find_waves: Precision = ', precision)
            print('find_waves: Returning BAD LP: Could not get enough bounds to construct LP')
            return 'BAD LP'

        else:



            ##Now we need to find a basis for the rows of this includes rows Gamma2
            ### (actually a basis for the row space of Gamma2) and such that
            ### if we solve Bv = bddts[B], then we get A_jv \leq bddts[j]
            #### We can get that by solving a LP



            if report_activity:
                try:
                    flobj.write("find_waves: Removing/Adding Constraints\n")
                except:
                    print("find_waves: Removing/Adding Constraints")
                taddrmv = time.time()

            constrnames = {} #Name of constraint to index in my problem
            for ii in range(MatA.shape[0]):
                constrnames['LE[' + str(ii) + ']'] = ii
            for jj in range(Gamma2.shape[0]):
                constrnames['Kernal[' + str(jj) + ']'] = jj + MatA.shape[0]

            dontrmv = ['LE[' + str(i) + ']' for i in atthem[0]]

            if solver == 'gb':

                for nm in inthere:
                    findbasis.getConstrByName(nm).RHS = bddts[constrnames[nm]]

                findbasis.update()



                findbasis.remove([con for con in [findbasis.getConstrByName(conNm) for conNm in inthere if conNm not in dontrmv]])
                findbasis.update()



                findbasis.addConstrs((gb.quicksum([MatA[i][l]*vdots[l] for l in range(len(vdots))]) <= bddts[i] for i in atthem[0] if (not ('LE[' + str(i) + ']' in inthere))), name = 'LE')

                findbasis.update()

                inthere = [con.ConstrName for con in [findbasis.getConstrByName('LE[' + str(i) + ']') for i in range(len(bddts))] if con != None]


                if report_activity:
                    try:
                        flobj.write("find_waves: Added new constraints in " + str(time.time()-taddrmv) + " seconds.\n")
                        flobj.write("find_waves: optimizing with " + str(findbasis.NumConstrs) + " constraints\n" )
                    except:
                        print("find_waves: Added new constraints in ",time.time()-taddrmv, " seconds.")
                        print("find_waves: optimizing with ",findbasis.NumConstrs," constraints" )

            elif solver == "coin":

                findbasis.addConstraint(np.matrix(the_rows_of_A) * vdots <= bddts[atthem], name = "LE")

                if report_activity:
                    try:
                        flobj.write("find_waves: Added new constraints in " + str(time.time()-taddrmv) + " seconds.\n")
                        flobj.write("find_waves: optimizing with " + str(findbasis.nConstraints) + " constraints\n" )
                    except:
                        print("find_waves: Added new constraints in ",time.time()-taddrmv, " seconds.")
                        print("find_waves: optimizing with ",str(findbasis.nConstraints)," constraints" )


            else:
                for nm in inthere:
                    findbasis.linear_constraints.set_rhs(nm,bddts[constrnames[nm]])
                findbasis.linear_constraints.delete([con for con in inthere if con not in dontrmv])

                needtoadd = [at for at in atthem[0] if (not ('LE[' + str(at) + ']' in inthere))]

                newconstrs = [[vdots,list(MatA.astype(float)[rw])] for rw in needtoadd]
                new_rhs = [bddts[i] for i in needtoadd]
                new_names = ['LE[' + str(i) + ']' for i in needtoadd]

                findbasis.linear_constraints.add(lin_expr = newconstrs, senses = ['L']*len(newconstrs), rhs = new_rhs, names = new_names)

                inthere =  [nm for nm in findbasis.linear_constraints.get_names() if 'LE' in nm]




                if report_activity:
                    try:
                        flobj.write("find_waves: Added new constraints in " + str(time.time()-taddrmv) + " seconds.\n")
                        flobj.write("find_waves: optimizing with " + str(findbasis.linear_constraints.get_num()) + " constraints\n" )
                    except:
                        print("find_waves: Added new constraints in ",time.time()-taddrmv, " seconds.")
                        print("find_waves: optimizing with ",findbasis.linear_constraints.get_num()," constraints" )



            if solver == 'gb':
                findbasis.optimize()

                findbasis.update()


                status = findbasis.status


                statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}

                if report_activity:
                    if status in statusdic.keys():
                        try:
                            flobj.write("find_waves: LP Status: " + statusdic[status] + '\n')
                        except:
                            print("find_waves: LP Status: ", statusdic[status])
                    else:
                        try:
                            flobj.write("find_waves: LP Status: Other")
                        except:
                            print("find_waves: LP Status: Other")




                if statusdic[status] == "INFEASIBLE":
                    if report_activity:
                        try:
                            flobj.write('infeasible, will need to lower step size\n')
                        except:
                            print('infeasible, will need to lower step size')
                    return 'infeasible'


                if statusdic[status] == "INF_OR_UNBD":
                    findbasis.reset()
                    findbasis.Params.DualReductions = 0
                    findbasis.optimize()

                    status = findbasis.status

                    if report_activity:
                        if status in statusdic.keys():
                            try:
                                flobj.write("find_waves: LP Status: " + statusdic[status] + '\n')
                            except:
                                print("find_waves: LP Status: ", statusdic[status])
                        else:
                            try:
                                flobj.write("find_waves: LP Status: Other")
                            except:
                                print("find_waves: LP Status: Other")


                    if statusdic[status] == "INFEASIBLE":
                        return 'infeasible'


                if statusdic[status] == "UNBOUNDED":
                    if precision > minprecision:
                        findbasis.printStats()
                        precision = precision - 1
                    else:
                        print('find_waves: Precision = ', precision)
                        findbasis.printStats()
                        print('find_waves: Returning BAD LP: Could not get enough basis vectors')
                        return 'BAD LP'


                if (statusdic[status] != "OPTIMAL") and (statusdic[status] != "UNBOUNDED"):
                    # print('STATBAD')
                    print('find_waves: Precision = ', precision)
                    findbasis.printStats()
                    print('find_waves: Returning BAD LP - ', statusdic[status])
                    return 'BAD LP'

            elif solver == 'coin':

                findbasis.initialSolve()

                status = findbasis.getStatusCode()


    # -1 - unknown e.g. before solve or if postSolve says not optimal
    # 0 - optimal
    # 1 - primal infeasible
    # 2 - dual infeasible
    # 3 - stopped on iterations or time
    # 4 - stopped due to errors
    # 5 - stopped by event handler (virtual int ClpEventHandler::event())
    #
    #

                statusdic = {-1:"UKNOWN", 0:"OPTIMAL", 1:"INFEASIBLE",2:"INFEASIBLE",3:"ITERATIONSTIME",4:"ERRORS",5:"EVENT"}

                if report_activity:
                    if status in statusdic.keys():
                        try:
                            flobj.write("find_waves: LP Status: " + statusdic[status] + '\n')
                        except:
                            print("find_waves: LP Status: ", statusdic[status])
                    else:
                        try:
                            flobj.write("find_waves: LP Status: Other")
                        except:
                            print("find_waves: LP Status: Other")




                if statusdic[status] == "INFEASIBLE":
                    if report_activity:
                        try:
                            flobj.write('infeasible, will need to lower step size\n')
                        except:
                            print('infeasible, will need to lower step size')
                    return 'infeasible'

                #

                elif statusdic[status] != "OPTIMAL":
                    # print('STATBAD')
                    print('find_waves: Precision = ', precision)
                    findbasis.printStats()
                    print('find_waves: Returning BAD LP - ', statusdic[status])
                    return 'BAD LP'
            else:
                findbasis.solve()

                statusdic = {1:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",2:"UNBOUNDED"}
                status = findbasis.solution.get_status()

                if report_activity:
                    if status in statusdic.keys():
                        try:
                            flobj.write("find_waves: LP Status: " + statusdic[status] + '\n')
                        except:
                            print("find_waves: LP Status: ", statusdic[status])
                    else:
                        try:
                            flobj.write("find_waves: LP Status: Other")
                        except:
                            print("find_waves: LP Status: Other")

                if statusdic[status] == "INFEASIBLE":
                    if report_activity:
                        print('infeasible, will need to lower step size')
                    return 'infeasible'

                if statusdic[status] == "INF_OR_UNBD":
                    checkobj = [[vdots,list(obje)]]
                    findbasis.linear_constraints.add([checkobj],senses = ['L'],rhs = [1], names = ['check'])
                    findbasis.solve()
                    chckstatus = findbasis.solution.get_status()
                    if chckstatus == 1:
                        status = 2
                    else:
                        status = 3

                    if report_activity:
                        if status in statusdic.keys():
                            try:
                                flobj.write("find_waves: LP Status: " + statusdic[status] + '\n')
                            except:
                                print("find_waves: LP Status: ", statusdic[status])
                        else:
                            try:
                                flobj.write("find_waves: LP Status: Other")
                            except:
                                print("find_waves: LP Status: Other")


                    if statusdic[status] == "INFEASIBLE":
                        return 'infeasible'

                if statusdic[status] == "UNBOUNDED":
                    if precision > minprecision:
                        findbasis.printStats()
                        precision = precision - 1
                    else:
                        print('find_waves: Precision = ', precision)
                        print('find_waves: Returning BAD LP: Could not get enough basis vectors')
                        return 'BAD LP'

                if (statusdic[status] != "OPTIMAL") and (statusdic[status] != "UNBOUNDED"):
                    print('find_waves: Precision = ', precision)
                    print('find_waves: Returning BAD LP - ', statusdic[status])
                    return 'BAD LP'




            if statusdic[status] == "OPTIMAL":
                LP_not_ready = False
                allbddts = np.concatenate([bddts,np.zeros(len(Gamma2))])

                allem = np.concatenate([atthem[0],np.arange(len(MatA),len(MatA)+len(Gamma2))])
                #

                sparmsnames = {'s' + str(i):i for i in range(MatA.shape[1])}#variable name to variable index in MatA
                if solver == 'gb':
                    vard = {v.getAttr(gb.GRB.Attr.VarName):v for v in findbasis.getVars()}#variable name to variable object
                    vardict = {vard[v]:sparmsnames[v] for v in sparmsnames.keys()}#variable object to variable index in MatA

                    rws = [(c.getAttr(gb.GRB.Attr.ConstrName),get_expr_coos(findbasis.getRow(c),vardict),c.getAttr(gb.GRB.Attr.Slack),c.getAttr(gb.GRB.Attr.RHS)) for c in findbasis.getConstrs() if c.getAttr(gb.GRB.Attr.CBasis) == -1]#tuples of row name, row, of basic rows (i.e. non-basic slacks)

                    the_basis_i_need = np.array([r[1] for r in rws])#non-basic constraints.



                    the_index_of_basis = [constrnames[r[0]] for r in rws]

                elif solver == 'coin':

                    the_index_of_basis_init = np.where(findbasis.getStatusArray()[MatA.shape[1]:] != 1)# non-basic slacks

                    the_basis_i_need = np.concatenate([Gamma2,the_rows_of_A],axis = 0)[the_index_of_basis_init]

                    if the_basis_i_need.shape[0]<the_basis_i_need.shape[1]:
                        stop = False
                        ## If we don't have enough basis vectors (b/c not all variables are basic), one possible solution
                        ## is to take all the constraints that our solution intersects (i.e. all zero slacks) and simply choose
                        ## a basis from these (using QR decomposition).
                        while not stop:
                            slkvals = np.concatenate([findbasis.primalConstraintSolution["Kernal"],findbasis.primalConstraintSolution["LE"]])
                            zeroslks = np.where(slkvals.round(precision) == 0)
                            allat = np.concatenate([Gamma2,the_rows_of_A],axis = 0)[zeroslks]

                            if np.linalg.matrix_rank(allat) == the_basis_i_need.shape[1]:
                                _,_,piv = sp.linalg.qr(allat.T,pivoting = True)
                                li_cols = piv[:allat.shape[1]]
                                bas_ind_ar = zeroslks[0][li_cols]
                                bas_ind_ar.sort()
                                the_index_of_basis_init = (bas_ind_ar,)
                                the_basis_i_need = np.concatenate([Gamma2,the_rows_of_A],axis = 0)[the_index_of_basis_init]
                                stop = True

                            elif precision >  minprecision:
                                precision = precision - 1
                            ## If that doesn't work...
                            else:
                                print(allat.shape)
                                print(np.linalg.matrix_rank(allat))
                                print(np.concatenate([Gamma2,the_rows_of_A],axis = 0).shape)
                                print(np.linalg.matrix_rank(np.concatenate([Gamma2,the_rows_of_A],axis = 0)))
                                print(findbasis.primalConstraintSolution["LE"][np.where(findbasis.primalConstraintSolution["LE"].round(precision) != 0)])
                                stop = True

                    the_index_of_basis = (np.array([ii if ii < len(Gamma2) else atthem[0][ii-len(Gamma2)]+len(Gamma2) for ii in the_index_of_basis_init[0]]),)


                else:


                    # Returns a pair (head, x), where head is a list of variable indices and x is a list of floats indicating the values of those variables. Indices of basic slacks are specified by -rowindex - 1.
                    basic_head,basic_vals = findbasis.solution.basis.get_header()



                    allnms = findbasis.linear_constraints.get_names()

                    var_ord = findbasis.variables.get_names()
                    fullmat = np.array([findbasis.linear_constraints.get_coefficients([(cnm,s) for s in vdots]) for cnm in allnms])

                    nonbasvars = np.array([vdots[i] for i in range(fullmat.shape[1]) if i not in basic_head])#list of non-basic variables. need to put them in



                    all_slack_vals = np.zeros(len(allnms))
                    bslcks = []
                    for j in range(len(all_slack_vals)):
                        if -j-1 in basic_head:
                            all_slack_vals[j] = np.array(basic_vals)[np.where(np.array(basic_head) == -j-1)][0]
                            bslcks += [j]


                    #we may need to pivot in non-basic variables.
                    for ii in nonbasvars:
                        findbasis.advanced.pivot(ii,findbasis.advanced.no_variable,0)


                    basic_head,basic_vals = findbasis.solution.basis.get_header()
                    allnms = findbasis.linear_constraints.get_names()
                    #
                    var_ord = findbasis.variables.get_names()
                    #
                    nonbasvars = np.array([i for i in range(fullmat.shape[1]) if i not in basic_head])#list of non-basic variables. need to put them in
                    #
                    #
                    #
                    all_slack_vals = np.zeros(len(allnms))
                    bslcks = []
                    for j in range(len(all_slack_vals)):
                        if -j-1 in basic_head:
                            all_slack_vals[j] = np.array(basic_vals)[np.where(np.array(basic_head) == -j-1)][0]
                            bslcks += [j]


                    rws = [[allnms[i],findbasis.linear_constraints.get_coefficients([(allnms[i],s) for s in vdots])] for i in range(len(allnms)) if i not in bslcks]


                    the_basis_i_need = np.array([r[1] for r in rws])#non-basic constraints.



                    the_index_of_basis = [constrnames[r[0]] for r in rws]




                all_bddts = np.concatenate([bddts, np.zeros(len(Gamma2))])





                if np.linalg.matrix_rank(the_basis_i_need) < MatA.shape[1]:
                    if solver == 'gb':
                        if precision > minprecision and np.linalg.matrix_rank(np.concatenate([the_rows_of_A,Gamma2])) < MatA.shape[1]:
                            precision = precision - 1
                            LP_not_ready = True
                        else:
                            print('find_waves: Precision = ', precision)
                            print("find_waves: Returning BAD LP, not enough basis vectors")
                            return 'BAD LP'
                    else:
                        print("Missing dimensions?")
                        return 'BAD LP'



    if len(the_index_of_basis) > MatA.shape[1]:
        print('find_waves: Precision = ', precision)
        print("find_waves: Returning BAD LP, too many basis vectors.")
        return 'BAD LP'



    # print('find_waves: Done')
    if report_activity:
        try:
            flobj.write("find_waves: Found Basis, Computing QR Decomposition\n")
        except:
            print("find_waves: Found Basis, Computing QR Decomposition")
        for n in headsup:
            if n in the_index_of_basis:
                try:
                    flobj.write("find_waves: Constraint" + str(n) + " in basis.\n")
                except:
                    print("find_waves: Constraint", n," in basis.")
            else:
                if solver == 'gb':
                    modelvars = [findbasis.getVarByName('s' + str(i)) for i in range(MatA.shape[1])]
                    dvdtdotn = np.dot(MatA[n],np.array([v.x for v in modelvars]))
                else:
                    dvdtdotn = np.dot(MatA[n],np.array([findbasis.solution.get_values("s"+str(i)) for i in range(MatA.shape[1])]))
                try:
                    flobj.write("find_waves: Constraint " + str(n) + " - decreasing as " + str(bddts[n]) + " - dv/dt dot constraint n " + str(dvdtdotn) + "\n")
                except:
                    print("find_waves: Constraint ", n," - decreasing as ", bddts[n]," - dv/dt dot constraint n  ",dvdtdotn )


    QB,RB = np.linalg.qr(the_basis_i_need)
    if report_activity:
        try:
            flobj.write("find_waves: Done  in " + str(time.time() - t) +  " seconds.\n")
        except:
            print("find_waves: Done  in ",time.time() - t, " seconds.")

    return QB,RB,the_index_of_basis, findbasis
