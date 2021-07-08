from __future__ import print_function
from sympy import *
symbol_string = '''a,b,c,d,e,f,g,h,i,j,k,l,m,n,p,q,r,s,t,u,v,w,x,y,z,\
A,B,C,D,E,F,G,H,I,J,K,L,M,N,P,R,S,T,U,V,W,X,Y,Z,\
alpha,beta,gamma,delta,epsilon,eps,zeta,eta,theta,iota,\
kappa,lamda,mu,nu,xi,pi,rho,sigma,tau,phi,chi,psi,omega,\
rp,xhat,yhat,zhat'''
symbol_list = symbol_string.split(",")
for asdf in symbol_list:
    exec("{0} = symbols(\"{0}\",positive=True)".format(asdf))
import math as M
import matplotlib.pyplot as plt
import string
import numpy as np
init_printing(use_unicode=True) #pretty prints ASCII
from decimal import*
getcontext().prec = 60
from iteration_utilities import duplicates
from itertools import combinations as comb
import scipy
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import solve_bvp
from scipy.signal import savgol_filter
#use np.inf in quad function for bounds
from sympy import mathematica_code as mcode
import cmath as I
#1j is equal to sqrt(-1) or i
import pandas as pd
import astropy.io.fits as pyfits
import os
import time

def import_constants(imp_consts = True):
    if imp_consts:
        global G
        G = 6.67408e-11
        #G = 6.674e-11
        global g
        g = 9.80665
        #g = 9.81
        global Mole
        Mole = 6.022e23
        #Mole = 6.02e23
        global epsilon0
        epsilon0 = 8.8541878128e-12
        #epsilon0 = 8.854e-12
        global mu0
        mu0 = 1.25663706212e-6
        #mu0 = 1.257e-6
        global c
        c = 299792458
        #c = 3e8
        global k_EM
        k_EM = (4*M.pi*epsilon0)**-1
        global q
        q = 1.602176634e-19
        #q = 1.602e-19
        global MASS_PROTON
        MASS_PROTON = 1.672621898e-27
        #MASS_PROTON = 1.673e-27
        global MASS_ELECTRON
        MASS_ELECTRON = 9.10938356e-31
        #MASS_ELECTRON = 9.11e-31
        global MASS_NEUTRON
        mass_neutron = 1.674927471e-27
        #mass_neutron = 1.675e-27
        
        global EARTH
        global MASS
        global VALUE
        global UNITS
        global MOON
        global SUN
        global DISTANCE
        
        class EARTH:
            class MASS:
                VALUE = 5.97237e24
                UNITS = 'kilograms'
            class RADIUS:
                VALUE = 6.371e6
                UNITS = 'meters'
            def show():
                print(
f'''Mass = {EARTH.MASS.VALUE} {EARTH.MASS.UNITS}
Radius = {EARTH.RADIUS.VALUE} {EARTH.RADIUS.UNITS}''', \
                    end='')
        class SUN:
            class MASS:
                VALUE = 1.98855e30
                UNITS = 'kilograms'
            class RADIUS:
                VALUE = 6.95508e8
                UNITS = 'meters'
            def show():
                print(
f'''Mass = {SUN.MASS.VALUE} {SUN.MASS.UNITS}
Radius = {SUN.RADIUS.VALUE} {SUN.RADIUS.UNITS}''', \
                    end='')
        global amu_grams
        amu_grams = 1.6605e-30
        global amu_kg
        amu_kg = 1.6605e-27
        global AU
        AU = 1.495978707e11
        class MOON:
            class MASS:
                VALUE = 7.342e22
                UNITS = 'kilograms'
            class RADIUS:
                VALUE = 1.7374e6
                UNITS = 'meters'
            class DISTANCE:
                VALUE = 3.84402e8
                UNITS = 'meters'
            def show():
                print(
f'''Mass = {MOON.MASS.VALUE} {MOON.MASS.UNITS}
Radius = {MOON.RADIUS.VALUE} {MOON.RADIUS.UNITS}
Distance from Earth = {MOON.DISTANCE.VALUE} {MOON.DISTANCE.UNITS}''', \
                    end='')         
        global h_J
        h_J = 6.62607015e-34
        #h_J = 6.6261e-34
        global h_eV
        h_eV = 4.13566769692859e-15
        #h_eV = 4.136e-15
        global hbar_J
        hbar_J = h_J/(2*M.pi)
        global hbar_eV
        hbar_eV = h_eV/(2*M.pi)
        global Planck_Length
        Planck_Length = (hbar_J*G/c**3)**.5
        global a0
        a0 = 5.2917721067e-11 #Bohr Radius
        #a0 = 5.29e-11
        global atm_Pa
        atm_Pa = 101325

        global variable_name
        global units
        

        global k_B
        k_B = 1.38064853e-23 #Boltzmann constant
        global Boltzmann_Constant
        class Boltzmann_Constant:
            variable_name = 'k_B'
            units = 'Joules / Kelvin'
            def show():
                print(f'{Boltzmann_Constant.variable_name} = {k_B} {Boltzmann_Constant.units}')
        
        global g_e
        g_e = 2.0023193043617 #Gyromagnetic Ratio Electron
        global Gyromagnetic_Ratio_Electron
        class Gyromagnetic_Ratio_Electron:
            variable_name = 'g_e'
            units = 'Radians / ( Second * Tesla)'
            def show():
                print(f'{Gyromagnetic_Ratio_Electron.variable_name} = {g_e} {Gyromagnetic_Ratio_Electron.units}')
        
        global sigma_sb
        sigma_sb = 5.6703744e-08 # (M.pi**2 * k_B**4) / (60 * hbar_J**3 * c**2) #Stefan-Boltzmann constant
        global Stefan_Boltzmann_Constant
        class Stefan_Boltzmann_Constant:
            variable_name = 'sigma_sb'
            units = 'Kilograms / (Kelvin^4 * Seconds^3)'
            def show():
                print(f'{Stefan_Boltzmann_Constant.variable_name} = {sigma_sb} {Stefan_Boltzmann_Constant.units}')


    
#imp_consts = str(input('Import constants? (y/n) :'))
imp_consts = 'y'

if imp_consts == 'y':
    import_constants()


#insert constants from now on

def divergence():
    vector = input("Enter components of Vector separated by a comma: ")
    vector = vector.split(",")
    div = diff(vector[0],x) + diff(vector[1],y) + diff(vector[2],z)
    return div

def curl():
    row3 = input("Enter components of Vector separated by a comma: ")
    row3 = row3.split(",")
    for i in range(len(row3)):
        row3[i] = eval(row3[i])
    curl = xhat*(diff(row3[2],y) - diff(row3[1],z)) - yhat*(diff(row3[2],x) - diff(row3[0],z)) + zhat*(diff(row3[1],x) - diff(row3[0],y))
    return curl

def r_sqr(xdata,ydata,fn_name,guesses,bounds,maxfev=1E3):
    fn_curve = curve_fit(fn_name,np.nan_to_num(xdata),np.nan_to_num(ydata),p0=guesses,bounds=bounds,maxfev=int(maxfev))[0]
    res = np.array(ydata)-fn_name(xdata,*fn_curve)
    return [(1 - np.sum(res**2)/np.sum((np.array(ydata)-np.mean(np.array(ydata)))**2)),list(fn_curve)]


def plots(x = "", y = "", color = '', frmt = "", title_fontsize = 20, axes_fontsize = 12, xscale = "linear", yscale = "linear", xlabel = "", ylabel = "", suptitle = "", file_name = '', name_data = "", style = "plain", axis = "", scilimits = (-3,4)) :
    import string
    import matplotlib.pyplot as plt
    import math as M
    import numpy as np
    punct = set(string.punctuation)
    punct.remove(".")
        
    if frmt == "":
        frmt = str(input("Enter preferred format (ex. \"b-\" for blue connected line): "))
    elif frmt == ' ':
        frmt = ''
    if color == '':
        color = str(input('Enter preferred color (Hex, RGB Tuple, or name): '))
    elif color == ' ':
        color = ''
    if x == "":
        x = str(input("Enter data for x axis: "))
        x = x.split(",")
    if y == "":
        y = str(input("Enter data for y axis: "))
        y = y.split(",")
    if xlabel == "":
        xlabel = str(input("Enter x axis title: "))
    if ylabel == "":
        ylabel = str(input("Enter y axis title: "))
    if suptitle == "":
        suptitle = str(input("Enter Title of Graph: "))
    if name_data == "":
        name_data = str(input("Enter name for data (no legend if empty): "))
    elif name_data == ' ':
        name_data = ''
    if axis == "":
        axis = str(input("Enter \"x\", \"y\", or \"both\" for which axis to apply sci notation to: "))
    elif axis == ' ':
        axis = ''
    if style == 'sci':
        plt.ticklabel_format(style = style, axis = axis, scilimits = scilimits)
        xscale = 'linear'
        yscale = 'linear'
    plt.xlabel(xlabel, fontsize=axes_fontsize)
    plt.ylabel(ylabel, fontsize=axes_fontsize)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.suptitle(suptitle, fontsize=title_fontsize)
    if (frmt == '' and color == '') and name_data == '':
        plt.plot(x,y)
    elif (frmt == '' and color != '') and name_data == '':
        plt.plot(x,y,color=color)
    elif (frmt == '' and color == '') and name_data != '':
        plt.plot(x,y,label = name_data)
    elif (frmt == '' and color != '') and name_data != '':
        plt.plot(x,y,color=color,label = name_data)
    
    elif (frmt != '' and color == '') and name_data == '':
        plt.plot(x,y,frmt)
    elif (frmt != '' and color != '') and name_data == '':
        plt.plot(x,y,frmt,color=color)
    elif (frmt != '' and color == '') and name_data != '':
        plt.plot(x,y,frmt,label = name_data)
    elif (frmt != '' and color != '') and name_data != '':
        plt.plot(x,y,frmt,color=color,label = name_data)
    
    if name_data != '':
        plt.legend()
    
    if file_name == '':
        file_name = str(input('Enter desired file name: '))
    if file_name == '':
        print('Invalid input. File named \"test.png\" instead')
        file_name = 'test'
    
    plt.savefig("Plots/" + file_name + ".png")

def printProgressBar(iteration, total, rate, avg_rate, time_left, prefix = 'Progress', suffix = 'Complete', decimals = 3, length = 50, fill = "\u2588", printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:0" + str(decimals+4) + '.' + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if int(time_left/3600) != 0:
        time_left = time_left/3600
        print('\r{} |{}| {}% {}      rate = {:07.03f}      avg rate = {:07.03f}      Est. time left: {:.2f} hours     '.format(prefix, bar, percent, suffix, rate, avg_rate, time_left), end = printEnd)
    elif int(time_left/60) != 0:
        time_left = time_left/60
        print('\r{} |{}| {}% {}      rate = {:07.03f}      avg rate = {:07.03f}      Est. time left: {:.2f} minutes     '.format(prefix, bar, percent, suffix, rate, avg_rate, time_left), end = printEnd)
    else:               # The added stuff is the problem but why?
        print('\r{} |{}| {}% {}      rate = {:07.03f}      avg rate = {:07.03f}      Est. time left: {:.2f} seconds     '.format(prefix, bar, percent, suffix, rate, avg_rate, time_left), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


print('{function name}.__code__.co_varnames for argument names of function')
print('{function name}.__defaults__ for the default values of arguments in the function')
#print('CHANGED?')
#print('changed?')

print('print(__file__) gives file path to the file it\'s written in. Ex. ',__file__)
