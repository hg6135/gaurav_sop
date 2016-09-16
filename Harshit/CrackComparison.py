
# coding: utf-8

# In[4]:

#Packages to be used
from math import pi,factorial,log
import numpy as np
import scipy.integrate as integrate
from scipy import linalg as la
import scipy.special as spl
import matplotlib.pyplot as plt
from scipy import linalg as la
get_ipython().magic(u'matplotlib inline')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)
import numpy.polynomial.legendre as lgd
import numpy.polynomial.chebyshev as cheby


# In[11]:

#Harshit's method
a = 1           #Crack Half-Length
nu = 0.25       #Poisson's Ratio
G = 26.2        #Shear Stress
E = 2*G*(1+nu)  #Young's Modulus
ks=(1-2*nu)/(2*(1-nu)) #Plain Strain Condition

degree = 25             #degree of Chebyshev's Approximation
order = degree + 1      #order of Chebyshev's Approximation

#values of x to use for plot
X = np.linspace(-a,a,order+2)[1:order+1]

def First(x):         #For finding first term of LHS for given x
    first = np.zeros([order,])
    for n in xrange(order):
        first[n] = spl.eval_chebyu( n , x )
    return first


def Second(x):         #For finding second term of LHS for given x
    second = np.zeros([order,])
    for n in range(order):
        func = lambda z:(spl.eval_chebyt(n + 1,z))/(np.sqrt(1-z**2))
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

c_Cheby_1 = Chebysol(0.1)
xd = np.linspace(-1.0,1.0,1001)[1:1000]
cheby_approx = cheby.chebval(xd,c_Cheby_1)


# In[34]:

X = np.linspace(-a,a,order+2)[1:order+1]
a = 1           #Crack Half-Length
nu = 0.25       #Poisson's Ratio
G = 26.2        #Shear Stress
E = 2*G*(1+nu)  #Young's Modulus
ks=(1-nu)/(2) #Plain stress Condition

degree = 25            #degree of Chebyshev's Approximation
order = degree + 1      #order of Chebyshev's Approximation

def term1(i,x):
    return spl.eval_legendre(i,x)

def beta(i,j,k):
    if (i - 2*k - j) % 2 == 0:
        return 0.0
    else:
        return float(2*((-1)**k)*factorial(2*(i - k)))/((2**i)*factorial(k)*factorial(i - 2*k)*factorial(i - k)*(i - 2*k - j))
    
def term2(i,x):
    final = 0.0
    for k in range(int(i)/2 +1):
        for j in range(i - 2*k):
            integrand = lambda z:beta(i,j,k)*(z**j)/((a**2 - z**2)**0.5)
            final += integrate.quad(integrand,-a,x)[0]
    return final

def term3(i,x):
    func = lambda z: (spl.eval_legendre(i,x) * log((a+z)/(a-z)))/((a**2 - z**2)**0.5)
    return integrate.quad(func,-a,x)[0]

def lhs(i,x,alphas):
    if alphas == 0:
        return term1(i,x)
    else:
        return term1(i,x) + ((a**2 - x**2)**0.5)*alphas*(term2(i,x) - term3(i,x))/(pi*(1-ks))

def rhs():
    return (a**2 - X**2)/(2*(1-ks)*G)

def Legendre(alphas):
    LHS = np.zeros([order,order])
    for i in range(order):
        print i
        temp = np.zeros([order,])
        for j in range(order):
            temp[j] = lhs(j,X[i],alphas)
        LHS[i] = temp    
    RHS = rhs()
    C = la.solve(LHS,RHS)
    #Sol = np.zeros([order,])
    #for i in range(order):
     #   temp = 0.0
      #  for j in range(order):
       #     temp += C[j]*spl.eval_legendre(j,X[i])/((a**2 - X[i]**2)**0.5)
        #Sol[i] = temp
    return C

SolC_1 = Legendre(0.1)
phiC = lgd.legval(xd,SolC_1)
SolLeg1 = 2.0*phiC/np.sqrt(1-xd**2)


# In[17]:

#Anmol's method
#Function to be integrated
alpha = lambda i,k:(np.power(-1,k)*factorial(2*i-2*k))/(factorial(k)*factorial(i-k)*factorial(i-2*k))
beta = lambda i,j,k: (1.0/2**i)*alpha(i,k)*((1-np.power(-1,i-2*k-j))/(i-2*k-j))

def f_int(x,i):
    ret_val = 0.0
    for k in range(int(np.floor(i/2)) + 1):
        for j in range(0,i-2*k-1+1):
            ret_val += beta(i,j,k)*np.power(np.sin(x),j)
    
    ret_val = ret_val - spl.eval_legendre(i,np.sin(x))*np.log((1+np.sin(x))/(1-np.sin(x)))
    return ret_val

alphas = 0.1
nu = 0.25
k = np.sqrt((1-nu)/2)
G = 26.2
L1 = lambda x,alphas: (-np.sqrt(1-x**2)*alphas)/((1-np.power(k,2))*np.pi)
L2 = lambda x: (-np.sqrt(1-x**2))/(2.0*(1-np.power(k,2))*G)
def ci_coeff(x,i,alphas):
    return spl.eval_legendre(i,x) - L1(x,alphas)*integrate.quad(f_int,-np.pi/2.0,np.arcsin(x),args=(i))[0]

rhs = lambda x:-L2(x)*np.sqrt(1-x**2)

def Legsol(alphas,N):
    
    A = np.zeros(shape=(N,N))
    b = np.zeros(shape=(N,1))

    x_N = np.linspace(-1.0,1.0,N+2)[1:N+1]

    for i in range(N):
        #print i
        for j in range(N):
            A[i,j] = ci_coeff(x_N[i],j,alphas)
        b[i] = rhs(x_N[i])
    c = la.solve(A,b)
    return c
c_Leg = Legsol(0.1,26)
legendre_approx = lgd.legval(xd,c_Leg)


# In[39]:

plt.plot(xd,cheby_approx,'r',xd,(2.0*legendre_approx/np.sqrt(1-xd**2)).T,xd,SolLeg1,'g')


# In[35]:

SolLeg1.shape


# In[ ]:



