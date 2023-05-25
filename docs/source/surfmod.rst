The SurfMod Class
=====================

GSMs are converted from cobra format (or created) into this class for use with surfin_fba and MetConSIN. 

For dynamic FBA, it is convenient to rewrite the stoichiometric matrix of a GSM as 

.. math::

    \Gamma = \begin{bmatrix} I & \Gamma^* \\ 0 & \Gamma^{\dagger}\end{bmatrix}

We put the model in standard form by forming the system

.. math::

    \begin{bmatrix}
        \Gamma^*          & -\Gamma^*         & I & 0 & 0 & 0 & 0 &0 &0\\
        -\Gamma^*         & \Gamma^*          & 0 & I & 0 & 0 & 0 &0 &0\\
        I                 & 0                 & 0 & 0 & I & 0 & 0 &0 &0\\
        0                 & I                 & 0 & 0 & 0 & I & 0 &0 &0\\
        \Gamma^{\dagger}  & -\Gamma^{\dagger} & 0 & 0 & 0 & 0 & I &0 &0\\
        -\Gamma^{\dagger} & \Gamma^{\dagger}  & 0 & 0 & 0 & 0 & 0 &I &0\\
        \text{Forced}     & \text{On}         & 0 & 0 & 0 & 0 & 0 &0 &I\\
    \end{bmatrix}
    \begin{bmatrix}
        \vec{\psi}_f \\ \vec{\psi}_r \\ s
    \end{bmatrix}
    =
    \begin{bmatrix}
        f(y) \\ g(y) \\ \text{interior upper bounds} \\ \text{interior lower bounds} \\ 0 \\ 0 \\ \text{on values}
    \end{bmatrix}

where f,g are the upper/lower exchange bounds, and are functions of the
environmental metabolites.

Also, we don't care about :math:`\Gamma^{\dagger}` except for its kernal (see :doc:`method`), so we can replace :math:`\Gamma^{\dagger}` with a new
matrix with orthonormal rows that has the same kernal.

.. autoclass:: metconsin.surfmod.SurfMod
    :members:

.. Class Functions
.. ----------------

.. .. autofunction:: surfmod.SurfMod.fba_gb

.. .. autofunction:: surfmod.SurfMod.fba_clp

.. .. autofunction:: surfmod.SurfMod.findWave

.. .. autofunction:: surfmod.SurfMod.solve_phaseone

.. .. autofunction:: surfmod.SurfMod.solve_minmax

.. .. autofunction:: surfmod.SurfMod.compute_internal_flux

.. .. autofunction:: surfmod.SurfMod.compute_slacks
