import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Eq, solve
import pulp as p
from pulp import LpStatus
from scipy.spatial import ConvexHull


choice =input("do u want to maximize or minimize")
if choice=="maximize" or choice=="1" :
    Lp_prob = p.LpProblem('Problem', p.LpMaximize)
if choice == "minimize" or choice == "2":
    Lp_prob = p.LpProblem('Problem', p.LpMinimize)

X1 = p.LpVariable("X1")  # Create a variable x
Y1 = p.LpVariable("Y1")  # Create a variable y
print("enter the variable of the objective fonction")
A = float(input("enter first value"))
B = float(input("enter seconde value"))
#print("ur objective fonction is (%d)*x + (%d)*y" % (A, B))
Lp_prob += A * X1 + B* Y1
#  ---------------creating list that will contain user inputs (a ,b,c plus operator)--------------------
a_list=[]
b_list=[]
c_list=[]
operator_list=[]
graph_lim_abs=[]
graph_lim_cord=[]
#-------------------INPUT CONSTRAINTS----------------
print(" constraints  are in this form : a X1 + b X2<=>c")
cons = int(input("how many constraints u have including x,y>0"))
#print("u have %d constraints " % (cons))
for i in range(0, cons):
        print("constraint number %d:" %(i+1))
        z = float(input("enter first value"))
        a_list.append(z)
        w = float(input("enter seconde value"))
        b_list.append(w)
        p = float(input("entrer third value"))
        c_list.append(p)
        operator = input("Enter comparison operator: ")
        operator_list.append(operator)
        #print("constraint : (%d)*x + (%d)*y %s  %d" % (z, w, operator, p))
        #--------------------SETTING LIMITS OF THE GRAPH----------------------------------
        if w == 0:
            y15 = 0
            graph_lim_abs.append(y15)

        else:
            y15 = p / w
            graph_lim_abs.append(y15)
        if z == 0:
            x15 = 0
            graph_lim_cord.append(x15)
        else:
            x15 = p / z
            graph_lim_cord.append(x15)
    #---------------WILL NEED THIS FOR THE CBC SOLVER------------------
        if operator == "<=":
            Lp_prob += z * X1 + w * Y1 <= p
        elif operator == ">=":
            Lp_prob += z * X1 + w * Y1 >= p



ma= max(graph_lim_cord)
ma2= max(graph_lim_abs)
lG=max(ma, ma2)
#-------------------------finding Real corner points----------------------------------------------

real_points=[]
for i in range(0,cons-1):
    for j in range(i+1,cons):
            if a_list[i]==0 and b_list[j]==0:
               az=([c_list[j]/a_list[j],c_list[i]/b_list[i]])
               real_points.append(az)
               #print(real_points)
               break
            elif b_list[i]==0 and b_list[j]==0:
                if c_list[i]!=0:
                    az=c_list[i]/a_list[i]
                    real_points.append([az,0])

                elif c_list[j]!=0:
                    az=c_list[j]/a_list[j]
                    real_points.append([az,0])
                break
            elif a_list[i]==0 and a_list[j]==0:
                if c_list[i]!=0:
                    az=c_list[i]/b_list[i]
                    real_points.append([0, az])

                elif c_list[j]!=0:
                    az = c_list[j] / b_list[j]
                    real_points.append([0, az])
                break


            else  :
               x1, y1 = symbols('x1,y1')
              # defining equations
               eq1 = Eq((a_list[i]*x1 + b_list[i]*y1), c_list[i])

               eq2 = Eq((a_list[j]*x1 + b_list[j]*y1), c_list[j])

            # solving the equation
               #print("Values of 2 unknown variable are as follows:")

               solvee=(solve((eq1, eq2), (x1, y1)))
               az=list(solvee.values())
               real_points.append(az)
#print(real_points)
#print(len(real_points))
real_cr=[]
for j in range(len(real_points)):
    compteur = 0
    for i in range(0,cons):
        if operator_list[i] == "<=" :
          if ((a_list[i] * real_points[j][0]) + (b_list[i] * real_points[j][1])  <= c_list[i]) :
            compteur=compteur+1
        elif operator_list[i] == ">=" :
          if ((a_list[i] * real_points[j][0]) + (b_list[i] * real_points[j][1]) >= c_list[i]) :
              compteur=compteur+1
    if compteur == cons :
       real_cr.append(real_points[j])
#-------------devise the double list in two separed lists one for cords the other for abs-------------
cor=[]
abs=[]
for i in range(len(real_cr)):
   xx=real_cr[i][0]
   yy=real_cr[i][1]
   cor.append(xx)
   abs.append(yy)

#print(real_cr)
#print("-------------------------------------")
#print(cor)
#print(abs)
#------------------------------------------GRAPH--------------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(0,10,100)
plt.axis([-lG,lG , -lG, lG ])
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
color=["black","blue","green","red","yellow","orange","maroon"]
for i in range(0, cons):
   if b_list[i]==0 :
       plt.axvline(x=c_list[i]/a_list[i], label=("constraint number",i+1))
   else:
      plt.plot(x, (c_list[i]/b_list[i]) - (a_list[i]/b_list[i]) * x , color[i], label=("constraint number",i+1))


                                #---------fill between---------------



if len(cor)!= 0 :
     #plt.fill(cor, abs)
     pts = np.array(real_cr)
     hull = ConvexHull(pts)
     plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], 'red', alpha=0.5)

plt.grid(True)
plt.show()
#-----------------------------------PULP (CBC SOLVER) PART------------------------------
# print the linear problem
print(Lp_prob)
status=Lp_prob.solve()  # Solver
    #LpStatus[status]
#print(LpStatus[status])
stat=(LpStatus[status])
    # The solution status
if stat=='Optimal':
    print("X=",X1.varValue)
    print("Y=",Y1.varValue)
    sol = A * X1.varValue + B * Y1.varValue
    print("valeur de la fonction objective :", sol)

elif stat=='Unbounded':
    print("L’ensemble des solutions réalisables (admissibles) est un polyèdre convexe non borné, la solution optimale tend vers l'infini")
elif stat=='Infeasible':
    print("il n’existe pas de solution réalisable donc pas de solution optimale")
else :
    print("La solution réalisable n'a pas été trouvée (mais peut exister).")
