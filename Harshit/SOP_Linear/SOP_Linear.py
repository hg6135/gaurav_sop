#packages used

#packages to be downloaded
import numpy as np

import scipy.integrate as integrate
from scipy import linalg as la
import scipy.special as Cheby
from scipy.special import ellipe

import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)

#packages pre-installed
from math import pi
print 'Step 1'
#constants
a = 1           #Crack Half-Length
nu = 0.25       #Poisson's Ratio
G = 26.2        #Shear Stress
E = 2*G*(1+nu)  #Young's Modulus
ks=(1-2*nu)/(2*(1-nu)) #Plain Strain Condition

degree = 80             #degree of Chebyshev's Approximation
order = degree + 1      #order of Chebyshev's Approximation

#values of x to use for plot
X = np.linspace(-a,a,order+1,False)
X = np.delete(X,0)
Y1 = np.linspace(-0.9999,X[0],10,False)
Y2 = np.linspace(0.9999,X[-1],10,False)
X = np.concatenate((X,Y1,Y2), axis=0)
X.sort()
order += 20

def First(x):         #For finding first term of LHS for given x
    first = np.zeros([order,])
    for n in xrange(order):
        first[n] = Cheby.eval_chebyu( n , x )
    return first


def Second(x):         #For finding second term of LHS for given x
    second = np.zeros([order,])
    for n in range(order):
        func = lambda z:(Cheby.eval_chebyt(n + 1,z))/(np.sqrt(1-z**2))
        second[n] = integrate.quad(func,-1,x)[0]
    return second

#To find Chebyshev Approximation Coefficients for given alpha*
def Chebysol(alphas):
    A = np.zeros([order,order])
    
    B = (np.sqrt(np.power(a,2)-np.power(X,2)))/(2*G*(1-ks))
    
    for x in range(order):
        A[x] = First(X[x])

    for x in range(order):
        A[x] -= alphas*Second(X[x])/(1-ks)
    
    b = la.solve(A,B)
    return b

#To find value for given Chebyshev coefficients and x
def Chebysol_Mat(x,b):
    Sol_Mat = 0.0
    for i in range(order):
            Sol_Mat += b[i]*Cheby.eval_chebyu( i , x )
    return Sol_Mat

def Persol(alphas):     #To find Perturbation Solution for given alpha
    per = np.zeros([order,])
    for i in range(order):
        x = X[i]
        func = lambda t : (t * ellipe(t**2)/((t**2-x**2)**0.5)) 
        ans0=integrate.quad(func,abs(x),1)[0]
        ans = 4*alphas*(1-nu)*ans0/pi
        ans1 = 2*(1-nu**2)*((1-x**2)**0.5 - ans)/E
        per[i] = 2*ans1
    return per
print 'Step 2'
#Classical Solution i.e alpha* = 0
b_Classical = Chebysol(0)
Classical_Mat = np.zeros([order,])
for i in range(order):
    Classical_Mat[i] = Chebysol_Mat(X[i],b_Classical)
print 'Step 3'
#Chebyshev Solution for alpha* = 0.1
b_Cheby_1 = Chebysol(0.1)
Cheby_Mat_1 = np.zeros([order,])
for i in range(order):
    Cheby_Mat_1[i] = Chebysol_Mat(X[i],b_Cheby_1)
print 'Step 4'
#Chebyshev Solution for alpha* = 0.2
b_Cheby_2 = Chebysol(0.2)
Cheby_Mat_2 = np.zeros([order,])
for i in range(order):
    Cheby_Mat_2[i] = Chebysol_Mat(X[i],b_Cheby_2)
print 'Step 5'
#Chebyshev Solution for alpha* = 0.5
b_Cheby_3 = Chebysol(0.5)
Cheby_Mat_3 = np.zeros([order,])
for i in range(order):
    Cheby_Mat_3[i] = Chebysol_Mat(X[i],b_Cheby_3)
print 'Step 6'
#Perturbation Solutions for above 3 alpha*
Per_Mat_1 = Persol(0.1)
Per_Mat_2 = Persol(0.2)
Per_Mat_3 = Persol(0.5)
print 'Step 7'
#Graph Plot for Classical ,Cohesive and Perturbation
fig1 = plt.figure(num=1, figsize=(20, 10),
                  dpi=50, facecolor='w', edgecolor='k')
ax1 = plt.subplot(111)
ax1.set_xlabel('x/a')
ax1.set_ylabel('normalised crack aperture 2*v(x)/a')
ax1.plot(X,2*Classical_Mat,'s',color='k',label="Classical")
ax1.plot(X,2*Cheby_Mat_1,'-',color='k',label="Cohesive")
ax1.plot(X,Per_Mat_1,'--',color='k',label="Perturbation")

ax1.plot(X,2*Cheby_Mat_2,'-',color='k')
ax1.plot(X,Per_Mat_2,'--',color='k')

ax1.plot(X,2*Cheby_Mat_3,'-',color='k')
ax1.plot(X,Per_Mat_3,'--',color='k')

ax1.annotate(r'$\alpha^* = 0.1$',xy=(X[30],Per_Mat_1[30]), xytext=(X[50],
        0.035),arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))
ax1.annotate('',xy=(X[70],2*Cheby_Mat_1[70]), xytext=(X[54],0.0365),
             arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))

ax1.annotate(r'$\alpha^* = 0.2$',xy=(X[25],Per_Mat_2[25]), xytext=(X[50],
        0.025),arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))
ax1.annotate('',xy=(X[75],2*Cheby_Mat_2[75]), xytext=(X[54],0.0265),
             arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))

ax1.annotate(r'$\alpha^* = 0.5$',xy=(X[20],Per_Mat_3[20]), xytext=(X[50],
        0.010),arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))
ax1.annotate('',xy=(X[80],2*Cheby_Mat_3[80]), xytext=(X[54],0.0113),
             arrowprops=dict(width = 2,headwidth = 10,facecolor='black'))

plt.legend(loc=1)
fig1.savefig("Result.png")
print 'Step 8 (Result Image Saved)'
#For plotting error graph
alp = np.zeros([100,])  #different alpha* matrix
C = Classical_Mat[50]

for i in range(1,51):
    alp[i-1] = 1.0/(i*2)
for i in range(51,101):
    alp[i-1] = float(i)

alp.sort()
print 'Step 9'
#Finding Error for given alpha*
err = np.zeros([100,])
for i in range(100):
    b_temp = Chebysol(alp[i])
    P = Persol(alp[i])[50]
    err[i] = (2*Chebysol_Mat(X[50],b_temp) - P)/(2*C - P)
print 'Step 10'
#Graph Plot for error vs 1/alpha*
fig2 = plt.figure(num=2, figsize=(20, 10),
                  dpi=50, facecolor='w', edgecolor='k')
ax2 = plt.subplot(111)
ax2.set_xlabel(r'$1/\alpha^*$')
ax2.set_ylabel(r'$Error(Correction Terms)_(x=0)$')
ax2 = plt.subplot(111)
ax2.plot(1/alp,err)
fig2.savefig("Error.png")
print 'Step 11 (Error Image Saved)'
X_2 = np.linspace(1,18,order+1,False)
X_2 = np.delete(X_2,0)

#Finding values of stress for classical solution outside crack
C_stress = np.zeros([order,])
for i in range(order):
    C_stress[i] = (X_2[i])/(X_2[i]**2 - 1)**0.5
print 'Step 12'
#Finding values of stress outside crack
stress_1 = np.zeros([order,])
for i in range(order):
    temp = 0.0
    for n in range(order):
        func = lambda z:b_Cheby_1[n]*Cheby.eval_chebyu(n,z)
                        *((1 - z**2)**0.5)/(z - X_2[i])
        temp += integrate.quad(func,-1,1)[0]
    stress_1[i] = (2*0.15*G*temp/pi + X_2[i])/(X_2[i]**2 - 1)**0.5
print 'Step 13'
stress_2 = np.zeros([order,])
for i in range(order):
    temp = 0.0
    for n in range(order):
        func = lambda z:b_Cheby_1[n]*Cheby.eval_chebyu(n,z)
                        *((1 - z**2)**0.5)/(z - X_2[i])
        temp += integrate.quad(func,-1,1)[0]
    stress_2[i] = (2*0.8*G*temp/pi + X_2[i])/(X_2[i]**2 - 1)**0.5
print 'Step 14'
#Chebyshev Solution for alpha* = 0.15
b_Cheby_4 = Chebysol(0.15)
Cheby_Mat_4 = np.zeros([order,])
for i in range(order):
    Cheby_Mat_4[i] = Chebysol_Mat(X[i],b_Cheby_4)
print 'Step 15'
#Chebyshev Solution for alpha* = 0.8
b_Cheby_5 = Chebysol(0.8)
Cheby_Mat_5 = np.zeros([order,])
for i in range(order):
    Cheby_Mat_5[i] = Chebysol_Mat(X[i],b_Cheby_5)
print 'Step 16'    
#Graph Plot for stress with alpha* = 0.15
fig3 = plt.figure(num=3, figsize=(20, 10), dpi=50,
                  facecolor='w', edgecolor='k')
ax3 = plt.subplot(111)
ax3.set_xlabel('x/a')
ax3.set_ylabel(r'Stress $\sigma_y$ in crack plane')
ax3.plot(X_2,stress_1,'o',color='k',label="Classical")
ax3.plot(X_2,C_stress,'--',color='k',label="Stress")
ax3.plot(X,4*0.15*G*Cheby_Mat_4,color='k',label="Cohesive")
plt.legend(loc=1)
plt.text(18.5, 0.1,r'$\alpha^* = 0.15$', size=20,ha="center",
         va="center",bbox = dict(boxstyle="square",
                                 ec=(0., 0., 0.),fc=(1., 1., 1.)))
fig3.savefig("Stress_1.png")
print 'Step 17 (Stress Image 1 is saved)'
#Graph Plot for stress with alpha* = 0.8
fig4 = plt.figure(num=4, figsize=(20, 10), dpi=50,
                  facecolor='w', edgecolor='k')
ax4 = plt.subplot(111)
ax4.set_xlabel('x/a')
ax4.set_ylabel(r'Stress $\sigma_y$ in crack plane')
ax4.plot(X_2,stress_2,'o',color='k',label="Classical")
ax4.plot(X_2,C_stress,'--',color='k',label="Stress")
ax4.plot(X,4*0.8*G*Cheby_Mat_5,color='k',label="Cohesive")
plt.legend(loc=1)
plt.text(18.5, 0.1,r'$\alpha^* = 0.8$', size=20,ha="center",
         va="center",bbox = dict(boxstyle="square",
                                 ec=(0., 0., 0.),fc=(1., 1., 1.)))
fig4.savefig("Stress_2.png")
print 'Step 18 (Stress Image 2 Saved)'
print 'Completed'
