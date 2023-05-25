Method of MetConSIN and SurfinFBA
======================================

.. _dfba:

Dynamic FBA
---------------

Dynamic flux balance analysis is a constraint-based genome-scale medthod for simulating microbial growth based on the idea of flux balance analysis (see :cite:`lewis2012` for an overview of constraint based methods).
Flux balance analysis posits that reaction fluxes (:math:`\vec{v}`) of a microbe's metabolism mirror an optimal solution to the following linear program:

.. math::

    \text{maximize}(\vec{v} \cdot \vec{c})\\
    \text{subject to}:\\
    \Gamma \cdot \vec{v} = 0\\
    \vec{a} \leq \vec{v} \leq \vec{b}

where :math:`\vec{c}` represents a growth objective, :math:`\Gamma` is the matrix of the stoichiometry of the reactions of the metabolism (including both interanal and exchange reactions), and :math:`\vec{a}` and :math:`vec{b}`
represent some bounds on the possible flux of the reactions. The stoichiometric matrix :math:`\Gamma` can be written as 

.. math::

    \Gamma = \begin{bmatrix} I & \Gamma^* \\ 0 & \Gamma^{\dagger}\end{bmatrix}

making the optimization problem

.. math::

    \text{maximize}(\vec{\psi}\cdot \vec{\gamma})\\
    \Gamma^{\dagger} \vec{\psi} = 0\\
    \vec{c}^1(\vec{y}) \leq
    \Gamma^* 
    \vec{\psi} \leq \vec{c}^2(\vec{y})\\
    \vec{d}^1 \leq \vec{\psi}\leq \vec{d}^2

where now we have split the flux vector :math:`\vec{v}` into the two parts, internal reacions contained in :math:`\vec{\psi}` and exchange reactions (that exchange a metabolite with the environment) that can be found by multiplying 
:math:`\Gamma^* \vec{\psi}`. We also rewrite the constraints of the exchange reactions to have an explicit dependence on the available environmental metabolites :math:`\vec{y}`.

Dynamic FBA is based on the observation that solving for reaction fluxes using this linear program provides a growth rate for a microbe (:math:`\vec{\psi}\cdot\vec{\gamma}`) as well as the reaction rate for the reactions which consume
or produce environmental metabolites. This means that we can define a system of differential algebraic equations:

.. math::

    \frac{dx_i }{dt}= x_i (\vec{\gamma}_i\cdot \vec{\psi}_i)\\
    \frac{d\vec{y}}{dt} = -\sum_{i=1}^p x_i \Gamma^*_i \vec{\psi}_i

where :math:`\vec{\psi}` is determined at each time-point by a linear program from FBA. Note that each microbial taxa in the simulation gets its own FBA problem, and so the *only* coupling between taxa is from their
interactions with their shared metabolite pool (we do not assume any sort of "community" optimization as is done in other community FBA methods).

.. _surfinfba:

SurfinFBA
---------------

In :cite:`brunner2020minimizing`, we introduced a method for simulating dynamic FBA based on the observation that a basic optimal solution to FBA at an intial time-point provides a basis for the solution that can 
be re-used for future time-points. This means that there is some invertible matrix :math:`B` formed from the columns of the (reformatted) constraint matrix (see :doc:`surfmod` for details on how the problem is reformulated) 
corresponding index set so that

.. math::

    \vec{\psi}=B^{-1}\vec{c}^*(\vec{y})

is an optimal solution to the linear program (where :math:`\vec{c}^*` is a subset of the constraints corresponding to the relevant index set), and this will *remain* an optimal solution for some time interval, 
even as :math:`\vec{y}` changes. We showed that if we have an FBA solution, we can find the basis :math:`B` by solving a second optimization problem. This is carried out by :py:func:`findWave <metconsin.surfmod.SurfMod.findWave>`,
which attempts to maximize the length of the time interval on which a basis will give optimal solutions. Conveniently, the basis will give optimal solutions as long as it gives feasible solutions (which after we reformulate 
the problem simply means the fluxes are non-negative), so we can roughly estimate the time until a new basis is needed using the derivative of the constraints. 

The standard form of the FBA problem is

.. math::

    \text{minimize}(\vec{\gamma}^+ \cdot \vec{\psi}^+)\\
    \text{Subject to } A\vec{\psi}^+ = \vec{c}^+

where :math:`+` indicates that the vector is augmented with slacks, and A is formed from the stoichiometry matrix. This implies that

.. math::

    A\frac{d\vec{\psi}^+}{dt} = \frac{d\vec{c}^+}{dt}

which allows us to estimate the time until a variable in :math:`\vec{\psi}` becomes infeasible (negative). We therefore seek a basis
that provides a feasible solution to 

.. math::

    A\vec{\omega} = \frac{d\vec{c}^+}{dt}\\
    \omega_j >= 0 \text{ if } \psi_j = 0

such that the set of indices of the non-zero :math:`\psi` is a subset of the basic index set (these must be in the basis we choose). This is accomplished by
solving a phase-one problem in :py:func:`solve_phaseone <metconsin.surfmod.SurfMod.solve_phaseone>`.

Next, we can improve on our choice of basis by maximimizing the minimum (over the variables) estimated time to infeasibility. This is done by
:py:func:`solve_minmax <metconsin.surfmod.SurfMod.solve_minmax>`

Once we haved a basis for each taxa, the differential algebraic system becomes the system of ODEs

.. math::

    \frac{dx_i }{dt}= x_i (\vec{\gamma}_i\cdot B_i^{-1}\vec{c}^*_i(\vec{y})) \\
    \frac{d\vec{y}}{dt} = -\sum_{i=1}^p x_i \Gamma^*_i B_i^{-1}\vec{c}^*_i(\vec{y})

:py:func:`surfin_fba <metconsin.dynamic_simulation.surfin_fba>` works by computing and FBA solution for each taxa and calling :py:func:`findWave <metconsin.surfmod.SurfMod.findWave>` for each taxa to construct the above ODE.
It then simulates forward according to this ODE until some flux value for some taxa becomes infeasible. At that point, an optimal solution is already available from the forward simulation, so the method only needs to
call :py:func:`findWave <metconsin.surfmod.SurfMod.findWave>` to find a new basis for forward simulation. Simulation continues in this manner until a given stop time is reached or no such basis can be found (implying that the 
FBA problem itself has no valid solution).

.. _metconsin:

MetConSIN
------------

MetConSIN extends the idea of :ref:`surfinfba` with the observation that the ODE system created for smooth simulation for a time interval can be represented as a network of interactions between microbes and metabolites, as
well as an ermergen network of interactions between metabolites (as mediated by the microbes). We can rearrange the ODEs as

.. math:: 

    \frac{dx_i}{dt} =C_i x_i +  \sum_{j=1}^m a_{ij} x_i c_{ij}(y_j)  = x_i \left(C_i + \sum_{j=1}^m a_{ij} c_{ij}(y_j)\right)\\
    \frac{dy_l}{dt} = -\sum_{i=1}^p\left(D_{il}x_i + \sum_{j=1}^m b_{ijl}x_i c_{ij}(y_j)\right)


In this form, we can interpret the term :math:`a_{ij}x_i c_{ij}(y_j)` as edges from :math:`y_j` to :math:`x_i`, becuase they represent the effect of :math:`y_j` on the growth of :math:`x_i`. Additionally, the terms
:math:`D_{il}x_i` and :math:`b_{ijl}x_ic_{ij}(y_j)` represent the effect of :math:`x_i` on the available :math:`y_l` (e.g. production or consumption) and so can be interpreted as edges from :math:`x_i` to :math:`y_l`. This is
the basis of how :py:func:`species_metabolite_network <metconsin.make_network.species_metabolite_network>` creates a microbe-metabolite network. Notice that in the terms :math:`b_{ijl}x_ic_{ij}(y_j)`, the interacton we define between
:math:`x_i` and :math:`y_l` also involves :math:`y_j`. We say that :math:`y_j` mediates this interaction, and label the edge with :math:`y_j`.

The terms :math:`b_{ijl}x_ic_{ij}(y_j)` can also be interpreted as the effect of :math:`y_j` on :math:`y_l`, allowing :py:func:`species_metabolite_network <metconsin.make_network.species_metabolite_network>` to also create a
metabolite-metabolite network. 


.. bibliography:: reference.bib