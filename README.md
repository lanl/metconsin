# metconsin
**Met**abolic **Con**text **S**pecies **I**nteraction **N**etworks

The goal of this project is to generate microbial species interaction networks using constraint based metabolic modeling. To do this, MetConSIN seeks "conservation laws" in the dynamic FBA system. These will take the form of a piecewise constant function, implying local conservation from a which a local interaction network is implied.

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

MetConSIN seeks to create a pairwise species interaction network by finding a conservation law, meaning we seek matrices such that 

<img src="https://latex.codecogs.com/gif.latex?A_1&space;\frac{d\boldsymbol{x}}{dt}&space;&plus;&space;A_2&space;\frac{d\boldsymbol{y}}{dt}&space;=&space;0" title="A_1 \frac{d\boldsymbol{x}}{dt} + A_2 \frac{d\boldsymbol{y}}{dt} = 0" />

so that 

<img src="https://latex.codecogs.com/gif.latex?A_1&space;\boldsymbol{x}&space;&plus;&space;A_2&space;\boldsymbol{y}&space;=&space;\boldsymbol{\phi}_0" title="A_1 \boldsymbol{x} + A_2 \boldsymbol{y} = \boldsymbol{\phi}_0" />

and <img src="https://latex.codecogs.com/gif.latex?A_2" title="A_2" /> is invertible. This would allow us to write rewrite our system as 

<img src="https://latex.codecogs.com/gif.latex?\frac{dx_i}{dt}&space;=&space;x_i&space;\left[\boldsymbol{\chi}_i&space;\cdot&space;B_i^{-1}\boldsymbol{h}_i\left(&space;A_2^{-1}\left(\boldsymbol{\phi}_0&space;-&space;A_i&space;\boldsymbol{x}&space;\right&space;)&space;\right&space;)&space;\right&space;]" title="\frac{dx_i}{dt} = x_i \left[\boldsymbol{\chi}_i \cdot B_i^{-1}\boldsymbol{h}_i\left( A_2^{-1}\left(\boldsymbol{\phi}_0 - A_i \boldsymbol{x} \right ) \right ) \right ]" />

Finally, we may evaluate the function 

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\chi}_i&space;\cdot&space;B_i^{-1}\boldsymbol{h}_i\left(&space;A_2^{-1}\left(\boldsymbol{\phi}_0&space;-&space;A_i&space;\boldsymbol{x}&space;\right&space;)&space;\right&space;)" title="\boldsymbol{\chi}_i \cdot B_i^{-1}\boldsymbol{h}_i\left( A_2^{-1}\left(\boldsymbol{\phi}_0 - A_i \boldsymbol{x} \right ) \right )" />

to find metabolically contextualized species interactions. 
