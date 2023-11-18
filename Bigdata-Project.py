#!/usr/bin/env python
# coding: utf-8

# ###  Formulate and solve a linear program to determine the portfolio of stocks, bonds, and options that maximises expected profit

# ###   Importing the Gurobi Python module:

# In[1]:


import gurobipy as gp


# ###  Creating the model object:
# 
# 

# In[2]:


# Create model
model = gp.Model("portfolio")


# ###  Defining  decision variables :
# 
# - The variables are defined:
#      - B (number of bonds purchased)
#      - S (number of shares of stock XYZ purchased)
#      - C (number of call options purchased or sold).
# - We set the lower bound of B and S to 0 to ensure that we only buy assets, and set the upper and lower bounds of C to 50 and -50 respectively, to ensure that we do not buy or sell more than 50 call options.

# In[3]:


# Define decision variables

B = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="B")
S = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="S")
C = model.addVar(lb=-50, ub=50, vtype=gp.GRB.CONTINUOUS, name="C")



# ###   Set Objective function :
# 
# - This line sets the objective function of the model to maximize the expected profit, which is equal to 10*B + 4*S (10 for bonds and 4 for stock XYZ, as explained in the problem statement).
# - A 6 - month riskless zero coupon bond was purchased as 100 pounds but sold as 90 pounds.Thus Expected profit obtained as (100-90) = 10
# - Bonds: A 100 pound face value, six-month, risk-free, zero-coupon bond is sold for 90 pound. Therefore, B means that the expected return on a bond purchase is 10 pounds  and therefore 100-90 = 10.
# - Stock XYZ: The share price of Stock XYZ is currently 20 pounds. Here, the investor takes into account three possible outcomes for the price of stock XYZ in six months:
#      - Price will remain the same,resulting in no loss or profit
#      - Price increase to 40 pounds saying the investor would lose money because he sold for 20 pounds and it turned to 40 pounds, resulting in loss as  20-40 equals -20. We therefore regard this scenario to be 0 since it results in loss.
#      - Price dropping to 12 pounds implies that 12-20=-8
# - From the three situations above, it follows that the expected return on investment for stock XYZ as, which equals 1/3(20+ 0 -8)=12/3=4.
# - Option Call European call option bought 100 shares of stock XYZ at 15 pounds each, and after six months, it sold for 1000 pounds, which means it had originally cost 1500 pounds.
# - Similar results are obtained for the three scenarios that were previously considered: 1/3(1500-500-1000)=0.

# In[4]:


# Set objective
model.setObjective(10*B + 4*S, sense=gp.GRB.MAXIMIZE)


# ### Adding the budget constraint to the model:
# 
# - The first constraint, 90*B + 20*S + 1000*C <= 20000, limits the total amount of money that can be spent on purchasing bonds, stocks, and call options. The left-hand side of the inequality represents the total cost of the portfolio, while the right-hand side represents the available budget.
# 
# - The second constraint, C <= 50, sets a limit on the number of call options that can be purchased. The variable C represents the number of call options, and it is constrained to be less than or equal to 50.
# 
# - The third constraint, C >= -50, sets a limit on the number of call options that can be sold. The variable C is constrained to be greater than or equal to -50, which means that the investor can sell at most 50 call options.

# In[5]:


# Add constraints
model.addConstr(90*B + 20*S + 1000*C <= 20000, name="budget")
model.addConstr(C <= 50, name="call_option_purchase_limit")
model.addConstr(C >= -50, name="call_option_sale_limit")


# ###  Solving the model:
# 
# - Gurobi solver takes the model and solves it to find the optimal solution that maximizes or minimizes the objective function based on the constraints and variable bounds set in the model.

# In[6]:


# Optimize model
model.optimize()


# ### Printing the optimal solution:
# 
# - This line prints the optimal solution found by the solver, including the number of bonds purchased (B.x), the number of shares of stock XYZ purchased (S.x), the number of call options purchased or sold (C.x), and the expected profit (model.objVal).
# 
# - The optimal solution found by the linear programming model is to purchase 3500 shares of stock XYZ and to sell 50 call options, with no bonds purchased. This portfolio has an expected profit of £14000.0.

# In[7]:


# Print optimal solution and objective value
print("Optimal solution:")
print("--------------------")
print("B =", B.x)
print("S =", S.x)
print("C =", C.x)
print("Expected profit =", model.objVal)


# ###  II A (b)  Suppose that the investor wants a profit of at least £2,000 in any of the three scenarios for the price of XYZ six months from today. Formulate and solve a linear program that will maximise the investor’s expected profit under this additional constraint.
# 

# In[8]:


import gurobipy as gp

# create a new model
m = gp.Model()


# ### Decision Variables
# 
# - These lines of code define the decision variables for the optimization problem.
# - B, S, and C are continuous variables representing the amount of bonds, stocks, and call options purchased, respectively.
# - P1, P2, and P3 are also continuous variables representing the profits earned from three types of investments.
# - The lb argument specifies the lower bound of the variable, while ub specifies the upper bound. In this case, B and S are non-negative, so their lower bounds are set to zero. C is a call option purchase, so its value can be negative or positive, but it is limited between -50 and 50. 
# - All variables are continuous, meaning they can take on any value within their bounds. 
# - P1, P2, and P3 are the decision variables representing the profit in each scenario. So (1/3)*P1 + (1/3)*P2 + (1/3)*P3 calculates the expected profit.

# In[9]:


# add decision variables
B = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="B")
S = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="S")
C = m.addVar(lb=-50, ub=50, vtype=gp.GRB.CONTINUOUS, name="C")
P1 = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P1")
P2 = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P2")
P3 = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P3")

# set objective function
m.setObjective((1/3)*P1 + (1/3)*P2 + (1/3)*P3, sense=gp.GRB.MAXIMIZE)


# ### Adding constraints
# 
# - 90B + 20S + 1000C <= 20000: This represents a budget constraint that limits the amount of money that can be spent on buying stocks and call options to 20,000 pounds.
# - 10B + 20S + 1500C == P1: This constraint represents the expected profit from scenario 1, which involves the purchase of both stocks and call options.
# - 10B - 500C == P2: This constraint represents the expected profit from scenario 2, which involves the purchase of stocks only.
# - 10B - 8S - 1000C == P3: This constraint represents the expected profit from scenario 3, which involves the purchase of stocks and put options.
# - P1 >= 2000: This constraint ensures that the expected profit from scenario 1 is at least pounds 2000.
# - P2 >= 2000: This constraint ensures that the expected profit from scenario 2 is at least pounds 2000.
# - P3 >= 2000: This constraint ensures that the expected profit from scenario 3 is at least pounds 2000.
# 
# 

# In[10]:


# add constraints
m.addConstr(90*B + 20*S + 1000*C <= 20000)
m.addConstr(10*B + 20*S + 1500*C == P1)
m.addConstr(10*B - 500*C == P2)
m.addConstr(10*B - 8*S - 1000*C == P3)
m.addConstr(P1 >= 2000)
m.addConstr(P2 >= 2000)
m.addConstr(P3 >= 2000)


# ###  Solving and Printing the Model Solutions
# 
# - This code block first runs the optimization of the linear programming model using the m.optimize() method. Then, it uses the print() function to display the optimal solution values for each of the decision variables (B, S, C, P1, P2, P3) and the expected profit obtained from the solution (m.objVal).
# 
# - The f-string formatting is used to insert the values of the decision variables and the expected profit into the string output by the print() function.

# In[11]:


# optimize the model
m.optimize()
# print results
print("Optimal solution:")
print("--------------------")
print("B =", B.x)
print("S =", S.x)
print("C =", C.x)
print("P1 =",P1.x)
print("P2 =",P2.x)
print("P3 =",P3.x)
print("Expected profit =", m.objVal)


# - This output shows the optimal solution to a linear programming problem with decision variables B, S, C, P1, P2, and P3. The optimal solution found by the solver is B = 0.0, S = 2800.0, C = -36.0, P1 = 2000.0, P2 = 18000.0, P3 = 13600.0. The solver also found the expected profit to be 11,200 pounds.
# 
# - This indicates that the optimal solution to the problem involves buying 2800 units of S, selling 36 units of C, and not buying any units of B. The expected profit with this decision is 11,200 pounds , which is the maximum profit that can be obtained subject to the constraints imposed on the problem.
