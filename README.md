## REINFORCE
This repo constains the implementation of REINFORCE and REINFORCE-Baseline algorithm on Mountain car problem.

The result after training the agent with REINFORCE algorithm or REINFORCE-Baseline will look like this:

<p align="center">
<img src="./readme_gif/mountain_car_agent_eps7700.gif" width="488"/>
</p>


## Table of Contensts
* [What is REINFORCE algorithm?](#what-is-reinforce-algorithm)
* [Differentiable policy](#differentiable-policy)
    * [Tile coding](#tile-coding)
    * [Neural network](#neural-network)
### What is REINFORCE algorithm?

Proposed at 1992, REINFORCE is the root of policy gradient methods.  <br/>

In short, the algorithm estiamtes the return using monte carlo method and in return adjust the policy using gradient asent. <br/>

Below is the pseudo code:
<p align="center">
<img src="./readme_pic/REINFORCE.JPG" width="488"/>
</p>

A modified version on top of REINFORCE is REINFORCE-Baseline. <br/>

REINFORCE-baseline chooses the baseline using value estimate and subtracted from estimated return. The goal is to reduce variance. <br/>
<p align="center">
<img src="./readme_pic/REINFORCE_baseline.JPG" width="488"/>
</p>

### Differentiable policy

REINFORCE algorithm requires a differentiable policy parameterization. In theory, any function approxiamtion algorithm would work. <br/> 

For this implementation, I choose tile coding with linear approximation and softmax to keep things simple. <br/>
The policy is given as follows, <br/>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\pi&space;(a|s,\theta&space;)&space;=&space;\frac{e_{}^{h(s,a,\theta)}}{\sum_{b}^{}e_{}^{h(s,b,\theta)}}\&space;where\&space;h(s,a,\theta)&space;=&space;\theta_{}^{T}x(s,a)" title="\pi (a|s,\theta ) = \frac{e_{}^{h(s,a,\theta)}}{\sum_{b}^{}e_{}^{h(s,b,\theta)}}\ where\ h(s,a,\theta) = \theta_{}^{T}x(s,a)" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\pi(a|s,&space;\theta&space;)&space;=&space;\frac{e^{\theta&space;_{}^{T}x(s,c)-max\&space;\theta&space;_{}^{T}x(s,c)}}{\sum_{b}^{}e^{\theta&space;_{}^{T}x(s,b)-max\&space;\theta_{}^{T}x(s,c)}" title="\pi(a|s, \theta ) = \frac{e^{\theta _{}^{T}x(s,c)-max\ \theta _{}^{T}x(s,c)}}{\sum_{b}^{}e^{\theta _{}^{T}x(s,b)-max\ \theta_{}^{T}x(s,c)}" />
</p>

The gradient of the policy is, <br/>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\bigtriangledown&space;_{\theta&space;}ln\pi(a|s,\theta)&space;=&space;x(s,a)&space;-&space;E_{\pi&space;_{\theta&space;}}[x(s,a)]" title="\bigtriangledown _{\theta }ln\pi(a|s,\theta) = x(s,a) - E_{\pi _{\theta }}[x(s,a)]" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\bigtriangledown&space;_{\theta&space;}ln\pi(a|s,\theta)&space;=&space;x(s,a)&space;-&space;\sum_a&space;\pi&space;(a|s,&space;\theta&space;)x(s,a)" title="\bigtriangledown _{\theta }ln\pi(a|s,\theta) = x(s,a) - \sum_a \pi (a|s, \theta )x(s,a)" />
</p>

#### Tile coding

The feature x(s,a) is generated through tile coding. This method encodes the entire continous state space into binary vector. To take action into account, the encode vector needs to stack together depending on the number of actions.
<p align="center">
<img src="./readme_pic/feature_representation.JPG" width="488"/>
</p>
