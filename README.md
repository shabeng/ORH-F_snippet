# ORH-F_snippet
This repo has parts of my thesis code. 
My thesis concerns the assignment decisions of on-demand ride requests to vehicles in real-time within a ride-hailing system. The goal of the thesis was to develop a data-driven policy for the assignment decisions that will achieve efficiency and geographical fairness towards the passengers (denoted Z).    

This repo includes the following parts:
1. Creating the instances of the ride-hailing system, using the CityNetwork and ProblemObjects.
2. Solving the offline variation of the problem with one of my MILP models, using the CPLEX API for python using the SolutionsApproch/MILP. 
3. Solving the online problem with the dispatching rules, which were the benchmark in my thesis, and showing the results with a bar plot, using SolutionsApproch/Simulation and more. 

You could use the running files - solve_milp.py and solve_online_dispatch_rule.py, to run the different available parts.
