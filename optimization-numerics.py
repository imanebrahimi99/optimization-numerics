"""
By Ruolin Wang, Sebastian Westerlund, Ruth Risberg and Iman Ebrahimi, 2022-03-10

The three tasks are separated into different functions to put them in different scopes and make them run completely separately. All imports are collected at the top of the file and the three functions are called at the bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from scipy.optimize import fsolve
import scipy.integrate



ax = None # Task 2 uses a global ax

def Task1():
    # Task 1, Iman
    """
    Calculating the length of a plane curve using Riemann sums
 
    Def: Let "f(x)" be a bounded function defined on [a,b], then the sum:
    sigma(f(c_k)*delta(x_k)) for k = 1,2,...,n is called a Riemann Sum for f.
    where P(n) is a partition into N pieces on [a,b] 
    consisting of a set of points (x_k).
    c_k belongs to the Interval [x_k-1 , x_k] 
    for all k belonging to {1,2,...,N} 
 
    """
    
    #%%
    def Rie_sum(f,a,b,n):
        
        sigma = 0 
        delta_x = (b-a)/n
        
        for k in range(1,n):
            sigma += f(a+(k*delta_x))
        
        return delta_x * sigma 
        
    #Testing it with g(x) = x^2
    g = lambda x : x**2 
    
    print(f'Our Result = {Rie_sum(g,0,5,1000000)}')
    print(f'Exact Solution = {scipy.integrate.quad(g,0,5)}')
    
    #%%
    #Verifying that the difference tends to zero as n tends to infinity 
    
    infinity = 1000000
    exact_sol = scipy.integrate.quad(g,0,5)
    for n in range(1,10):
        difference = exact_sol[0] - Rie_sum(g,0,5,n)
        print(f'for n = {n}: difference = {difference}')
    
    
    difference = exact_sol[0] - Rie_sum(g,0,5,infinity)
    print(f'for n = {infinity}: difference = {difference}')
    #%%
    
    """
    calculating the length of the curve (x,y) = (t**2,t**3)
    """
    
    #v(t) = sqrt((2*t)**2 + (3*(t**2))**2)
    #simplified:
    v = lambda t: (np.abs(t))*(np.sqrt(4+(9*(t**2))))
    
    length = Rie_sum(v,-2,1,infinity)
    print(f'The length of the curve = {length}')
    #%%
    
    """
    #comparing the above results with scipy.integrate.quad:
    """
    print(f'The length of the curve according to our formula = {abs(length)}')
    exact_val = scipy.integrate.quad(v,-2,1)
    print(f'The length of the curve according to scipy = {exact_val[0]}')
    difference = exact_val[0] - length
    print(f'difference = {difference}')
    





def Task2():
    # Task 2, Ruolin
    
    global ax
    
    # %%
    def g(x):
        return 8*x[0]*x[1]-4*x[0]**2*x[1]-2*x[0]*x[1]**2+x[0]**2*x[1]**2
    def gm(x):
        return -(8*x[0]*x[1]-4*x[0]**2*x[1]-2*x[0]*x[1]**2+x[0]**2*x[1]**2)
    
    # %%
    pts = 251
    x_ = np.linspace(-10,10,pts)
    X, Y = np.meshgrid(x_, x_)
    Z = 8*X*Y-4*X**2*Y-2*X*Y**2+X**2*Y**2
    Z1 = -(8*X*Y-4*X**2*Y-2*X*Y**2+X**2*Y**2)
    fig, ax = plt.subplots()
    CS = ax.contour(X,Y,Z, levels=[-100, 0, 2, 200, 1000, 5000]) #Creates contour plot of g, levels added for visibility
    ax.clabel(CS, inline=True, fontsize=8) 
    ax.set_title('Contour plot of g(x)')
    #plt.show()
    
    #fig, ax = plt.subplots()
    #CS = ax.contour(X,Y,Z1, levels=[-5000, -1000, -200, -2, 0, 100])
    #ax.clabel(CS, inline=True, fontsize=8)
    #ax.set_title('Contour plot of -g(x)')
    #plt.show()
    
    
    # %% #Here we print the local maxes and mins. According to the contour plot, we expect the only place for local maxes/mins to be around (1,1)
    xmn = optimize.fmin(g,np.array([1,1]), xtol=0.00001, ftol=0.00001,maxiter=250)
    xmx = optimize.fmin(gm,np.array([1,1]), xtol=0.00001, ftol=0.00001,maxiter=250)
    print("Extrema: ",xmn,g(xmn),xmx,gm(xmx))
    
    # %%
    fig, ax = plt.subplots()
    def ret_func(x):
        global ax
        ax.scatter(x[0],x[1],s=5, edgecolors='none', c='green')
    CS = ax.contour(X,Y,Z1, levels=[-5000, -1000, -200, -2, 0, 100])
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_title('Contour plot of -g(x) with points tracing the iterations')
    xmx = optimize.fmin(gm,np.array([1,1]), xtol=0.00001, ftol=0.00001,maxiter=250,callback=ret_func) #"callback = ret_func" means ret_func is called on every iteration
    plt.show()

    
    """
    # %%
    fig, ax = plt.subplots()
    def ret_func(x):
        global ax
        ax.scatter(x[0],x[1],s=5, edgecolors='none', c='green')
    CS = ax.contour(X,Y,Z)
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_title('Default result presentation')
    xmx = optimize.fmin(g,np.array([0,0]), xtol=0.00001, ftol=0.00001,maxiter=50,callback=ret_func)
    plt.show()
    """









def Task3():
    # Task 3, Ruth and Sebastian
    
    res = 50     # res*res points will be plotted
    area = 10    # x- and y-values from -area to area will be plotted
    
    def numdf(f,a,b,diffs,h = 10**(-6)): #f is function, (a,b) is point and diffs is string (i.e "x", "y" or "xx"), h is stepsize
        if diffs == "x":
            return (1/(2*h)*(f(a+h,b)-f(a-h,b)))
    
        elif diffs == "y":
            return (1/(2*h)*(f(a,b+h)-f(a,b-h)))
        
        elif diffs == "xx":
            return (1/(h**2)*(f(a+h,b)-2*f(a,b)+f(a-h,b)))
    
        elif diffs == "yy":
            return (1/(h**2))*(f(a,b+h)-2*f(a,b)+f(a,b-h))
        
        elif diffs == "xy" or "yx":
            return ((1/(4*h**2)*(f(a+h,b+h)-f(a+h,b-h)-f(a-h,b+h)+f(a-h,b-h))))
    
    def f(x,y,z):
     return x + 2*y + z + np.exp(2*z) - 1
    
    
    
    x = []
    y = []
    
    
    ztrue = []
    
    def solz(x,y): #Numerically solves f(x,y,z) = 0 for given x,y
        def g(a): return f(x,y,a)
        return fsolve(g,5)[0] #5 as initial value works, 0 doesn't.
    
    for x0 in np.linspace(-area, area, res): #Creates ztrue list of numerically calculated z
        for y0 in np.linspace(-area, area, res):
            x.append(x0)
            y.append(y0)
            ztrue.append(solz(x0, y0)) 

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Numerical solutions for z")
    ax.scatter3D(x, y, ztrue) #Plot the nuerical solutions for z
    
    
    
    #Determine taylor polynomial
    Dx = numdf(solz,0,0,"x")
    Dy = numdf(solz,0,0,"y")
    Dxx = numdf(solz,0,0,"xx")
    Dyy = numdf(solz,0,0,"yy")
    Dxy = numdf(solz,0,0,"xy")
    
    def P2(x,y): return solz(0,0) + Dx * x + Dy *y + (1/2)*Dxx * x**2 + (1/2)*Dyy*y**2 + Dxy * x*y
    
    ztaylor = [] 
    for x0 in np.linspace(-area,area,res): #Creates ztaylor array of z through the taylor approximation
        for y0 in np.linspace(-area,area,res):
            ztaylor.append(P2(x0,y0))
    
    plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.set_title("Taylor polynomial")
    ax2.scatter3D(x,y,ztaylor) #Plot the taylor expansion
    
    
    
    #Creates zerror array of the error between the true and taylor z-values
    zerror = [abs(ztaylor[i]-ztrue[i]) for i in range(len(ztrue))]
    
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.set_title("Error")
    ax3.scatter3D(x, y, zerror) #plot the error
    
    plt.show()
    #Make plot of the taylor polynomial





# Call functions for each task
print("######## Task 1 ########")
Task1()
print("######## Task 2 ########")
Task2()
print("######## Task 3 ########")
Task3()
