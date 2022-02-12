## REINFORCE
This repo constains the implementation of REINFORCE and REINFORCE-Baseline algorithm on Mountain car problem.

The result after training the agent with REINFORCE algorithm or REINFORCE-Baseline will look like this:

<p align="center">
<img src="./readme_gif/mountain_car_agent_eps7700.gif" width="488"/>
</p>


## Table of Contensts
* [What is REINFORCE algorithm?](#what-is-reinforce-algorithm)



#### What is REINFORCE algorithm?

Proposed at 1992, REINFORCE is the root of policy gradient methods. 
In short, the algorithm estiamtes the return using monte carlo method and in return adjust the policy using gradient asent.

Below is the pseudo code:
<p align="center">
<img src="./readme_pic/REINFORCE.JPG" width="488"/>
</p>

A modified version on top of REINFORCE is REINFORCE-Baseline.
REINFORCE-baseline chooses the baseline using value estimate and subtracted from estimated return. The goal is to reduce variance.
<p align="center">
<img src="./readme_pic/REINFORCE_baseline.JPG" width="488"/>
</p>