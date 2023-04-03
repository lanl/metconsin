# metconsin
**Met**abolic **Con**text **S**pecies **I**nteraction **N**etworks

The goal of this project is to generate microbial interaction networks using constraint based metabolic modeling. To do this, MetConSIN uses the dynamic FBA system. 

## Installation
Clone from github. We plan to add pip install in the future

## Dependencies
MetConSIN requires [Gurobi](https://www.gurobi.com/documentation/9.5/) and gurobi's python package. Alternatively, MetConSIN can use the open source [CyLP](http://mpy.github.io/CyLPdoc/index.html), but this is much slower.

Metconsin also requires [numba](https://numba.pydata.org/) and [cobrapy](https://opencobra.github.io/cobrapy/).

## Documentation
Documentation is being written using Sphinx and saved in the docs folder. To compile the docs, install [Sphinx](https://www.sphinx-doc.org/en/master/index.html), navigate to the docs folder, and run "make html".

## How to use? 

Please see the DOCs. There's a *usage* page and a *tutorial* page. Quick start is:

	from metconsin import metconsin_sim,save_metconsin
	metconsin_return = metconsin_sim(community_members,model_info_file,**kwargs)
	save_metconsin(metconsin_return,"results")
	
You must provide a list of community members and a corresponding file indicating paths to metabolic models for those community members. *kwargs* include initial conditions, simulation length, simulation resolution, etc.

## Basis of the Method
Dynamic FBA can be written:

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{c}&space;\frac{dx_i}{dt}&space;=&space;x_i\left(\boldsymbol{\chi}_i&space;\cdot&space;\boldsymbol{v}_i&space;\right&space;)\\&space;\frac{d\boldsymbol{y}}{dt}&space;=&space;-\sum_i&space;x_i&space;\Gamma_i^*&space;\boldsymbol{v}_i&space;\end{array}" title="\begin{array}{c} \frac{dx_i}{dt} = x_i\left(\boldsymbol{\chi}_i \cdot \boldsymbol{v}_i \right )\\ \frac{d\boldsymbol{y}}{dt} = -\sum_i x_i \Gamma_i^* \boldsymbol{v}_i \end{array}" />

where each <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{v}" title="\boldsymbol{v}" /> maximizes

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\chi}_i&space;\cdot&space;\boldsymbol{v}_i" title="\boldsymbol{\chi}_i \cdot \boldsymbol{v}_i" />
subject to 
<img src="https://latex.codecogs.com/gif.latex?\left\{&space;\begin{array}{c}&space;\Gamma^{\dagger}_i&space;\boldsymbol{v}_i&space;=&space;0\\&space;v_{ij,min}&space;\leq&space;v_{ij}&space;\leq&space;v_{ij,max}\\&space;\tilde{v}_{ij,min}&space;\leq&space;\left(\Gamma^*_i\boldsymbol{v}_i&space;\right)_j&space;\leq&space;\kappa_{ij}y_j&space;\end{array}&space;\right\}" title="\left\{ \begin{array}{c} \Gamma^{\dagger}_i \boldsymbol{v}_i = 0\\ v_{ij,min} \leq v_{ij} \leq v_{ij,max}\\ \tilde{v}_{ij,min} \leq \left(\Gamma^*_i\boldsymbol{v}_i \right)_j \leq \kappa_{ij}y_j \end{array} \right\}" />

For any value of <img src="https://latex.codecogs.com/gif.latex?t_0,\boldsymbol{y}(t_0)" title="t_0,\boldsymbol{y}(t_0)" />, we can find a basis <img src="https://latex.codecogs.com/gif.latex?B_i" title="B_i" /> for each organism <img src="https://latex.codecogs.com/gif.latex?i" title="i" /> such that 

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{v}(t_0)&space;=&space;B_i^{-1}\boldsymbol{h}_i(\boldsymbol{y}(t_0))" title="\boldsymbol{v}(t_0) = B_i^{-1}\boldsymbol{h}_i(\boldsymbol{y}(t_0))" />

where <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{h}_i(\boldsymbol{y})" title="\boldsymbol{h}_i(\boldsymbol{y})" /> is a function of the constraints of the linear program and furthermore  

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{v}(t)&space;=&space;B_i^{-1}\boldsymbol{h}_i(\boldsymbol{y}(t))" title="\boldsymbol{v}(t) = B_i^{-1}\boldsymbol{h}_i(\boldsymbol{y}(t))" />

for <img src="https://latex.codecogs.com/gif.latex?t&space;\in&space;[t_0,t_1]" title="t \in [t_0,t_1]" /> for some <img src="https://latex.codecogs.com/gif.latex?t_1&space;>&space;t_0" title="t_1 > t_0" />.

We can then rewrite the system as 

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{c}&space;\frac{dx_i}{dt}&space;=&space;x_i\left(\boldsymbol{\chi}_i&space;\cdot&space;B^{-1}_i&space;\boldsymbol{h}_i(\boldsymbol{y})&space;\right&space;)\\&space;\frac{d\boldsymbol{y}}{dt}&space;=&space;-\sum_i&space;x_i&space;\Gamma_i^*&space;B^{-1}_i&space;\boldsymbol{h}_i(\boldsymbol{y})&space;\end{array}" title="\begin{array}{c} \frac{dx_i}{dt} = x_i\left(\boldsymbol{\chi}_i \cdot B^{-1}_i \boldsymbol{h}_i(\boldsymbol{y}) \right )\\ \frac{d\boldsymbol{y}}{dt} = -\sum_i x_i \Gamma_i^* B^{-1}_i \boldsymbol{h}_i(\boldsymbol{y}) \end{array}" />

for <img src="https://latex.codecogs.com/gif.latex?t&space;\in&space;[t_0,t_1]" title="t \in [t_0,t_1]" />, so that we have an ODE in that time interval.

MetConSIN interprets these ODEs as networks of interactions, which it builds for later analysis.

## References
James D. Brunner and Nicholas Chia. Minimizing the number of optimizations for efficient community dynamic flux balance analysis. PLOS Computational Biology, 16(9):1-20, 09 2020. doi: 10.1371/journal.  pcbi.1007786. [Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007786)

## TO DO (4/3/2023):

* pip installation
* Build in standard environments
    - Western
    - High Fiber
    - M9 media


## Future Plans:

Eventually, we plan to develop a rigorous species-species approximation of the ODE networks and characterize the error. 

An error-free species-species approximation is possible if we can find a conservation law, meaning we seek matrices such that 

<img src="https://latex.codecogs.com/gif.latex?A_1&space;\frac{d\boldsymbol{x}}{dt}&space;&plus;&space;A_2&space;\frac{d\boldsymbol{y}}{dt}&space;=&space;0" title="A_1 \frac{d\boldsymbol{x}}{dt} + A_2 \frac{d\boldsymbol{y}}{dt} = 0" />

so that 

<img src="https://latex.codecogs.com/gif.latex?A_1&space;\boldsymbol{x}&space;&plus;&space;A_2&space;\boldsymbol{y}&space;=&space;\boldsymbol{\phi}_0" title="A_1 \boldsymbol{x} + A_2 \boldsymbol{y} = \boldsymbol{\phi}_0" />

and <img src="https://latex.codecogs.com/gif.latex?A_2" title="A_2" /> is invertible. This would allow us to write rewrite our system as 

<img src="https://latex.codecogs.com/gif.latex?\frac{dx_i}{dt}&space;=&space;x_i&space;\left[\boldsymbol{\chi}_i&space;\cdot&space;B_i^{-1}\boldsymbol{h}_i\left(&space;A_2^{-1}\left(\boldsymbol{\phi}_0&space;-&space;A_i&space;\boldsymbol{x}&space;\right&space;)&space;\right&space;)&space;\right&space;]" title="\frac{dx_i}{dt} = x_i \left[\boldsymbol{\chi}_i \cdot B_i^{-1}\boldsymbol{h}_i\left( A_2^{-1}\left(\boldsymbol{\phi}_0 - A_i \boldsymbol{x} \right ) \right ) \right ]" />

Finally, we may evaluate the function 

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\chi}_i&space;\cdot&space;B_i^{-1}\boldsymbol{h}_i\left(&space;A_2^{-1}\left(\boldsymbol{\phi}_0&space;-&space;A_i&space;\boldsymbol{x}&space;\right&space;)&space;\right&space;)" title="\boldsymbol{\chi}_i \cdot B_i^{-1}\boldsymbol{h}_i\left( A_2^{-1}\left(\boldsymbol{\phi}_0 - A_i \boldsymbol{x} \right ) \right )" />

to find metabolically contextualized species interactions. 

However, such a conservation law is unlikely to exist, so we seek an approximation with desirable error characteristics.