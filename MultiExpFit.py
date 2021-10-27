import numpy as np
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt

def ExpFitDiffEvol(N, x, y):
    """
    Fits a multi-exponential decay by scipy.optimize.differential_evolution

    Parameters
    ----------
    N : integer
        number of exponentials.
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    red_chi_sq : float
        reduced chi squared as calculated by the sum of squared residuals
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    bounds = [[min(x), max(x)]]*N + [[0, 1000]]*N

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum(((y - np.dot(fi, np.exp(-np.outer(1./taui, x))))**2.)/(y))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    taui, fi = np.split(s, 2)
    return fi, 1./taui, red_chi_sq

def EvalExpFit(x, a, b):
    """
    Is used to evaluate multi-exponential function for parameter lists a, b

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    a : (N) array_like
        1-dimensional list of prefactors.
    b : (N) array_like
        1-dimensional list of exponents (positive).
    Returns
    -------
    y : (n) array
        y-values of the multi-exponential function corresponding to the x-values put in.
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.
    """
    return np.dot(a, np.exp(-np.outer(b, x)))

xData = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460]
yDataOld = [550
, 423, 328, 198, 219, 128, 79, 75, 51, 42, 43, 46, 29, 30, 19, 14, 12, 20, 11, 11, 19, 13, 18, 6, 15, 9, 13, 5, 15, 4, 7, 3, 7, 7, 10, 7, 5, 9, 4, 5, 4, 5, 6, 3, 5, 2]
yData = []
for i in yDataOld:
    yData.append(i-0.16)
unc = [23, 21, 18, 14, 15, 11, 9, 9, 7, 6, 7, 7, 5, 5, 4, 4, 3, 4, 3, 3, 4, 4, 4, 2, 4, 3, 4, 2, 4, 2, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1] 
newunc = []
for i in unc:
    newunc.append(i*np.sqrt(2.8))

a, b, red_chi_sq = ExpFitDiffEvol(2, xData, yData)
print(a)
print(b)
print(red_chi_sq)

yFit = EvalExpFit(xData, a, b)
residuals = np.subtract(yData, yFit)

#residual plot
fig = plt.figure()
plt.errorbar(xData, residuals, yerr=unc, fmt='ro')
plt.title('Residual Plot of Double Exponential Fit')
plt.xlabel('Time (s)')
plt.ylabel('Residual')
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
plt.close()

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yData, yerr=unc, fmt='ro')
plt.plot(xData, yFit, 'b', label = '744e^(-.0327t) + 30.2e^(-.00492t)')
plt.title('Double Exponential Fit vs Detection Event Counts')
plt.xlabel('Time (s)')
plt.ylabel('Count Within 10 Second Interval')
plt.legend()
plt.savefig('data.png', dpi=300, bbox_inches='tight')
plt.close()

xx = np.linspace(0,65,num=65)
prodyFit110 = EvalExpFit(xx, [1], [.03272101])
prodyFit108 = EvalExpFit(xx, [1], [.00491965])
prod110 = []
prod108 = []
for i in prodyFit110:
    prod110.append(1097.47*(1-i))
for i in prodyFit108:
    prod108.append(114.767*(1-i))
prodtot = []
for i in range (0, 65):
    prodtot.append(prod110[i]+prod108[i])

shiftxData = []
for i in xData:
    shiftxData.append(i+73)
extxData = []
for i in xData:
    extxData.append(i)
extxData.insert(0,0)
extxData.insert(0, -8)
extyFit110 = EvalExpFit(extxData, [743.84649531], [.03272101])
extyFit108 = EvalExpFit(extxData, [30.23425077], [.00491965])
extyFit = EvalExpFit(extxData, a, b)
shiftextxData = []
for i in extxData:
    shiftextxData.append(i+73)

#plot of separate production and decay regions as well as total vs. measured data
fig = plt.figure()
plt.errorbar(shiftxData, yData, yerr=unc, fmt='ro', elinewidth=1, ms=3)
plt.plot(xx, prod110, 'b', label = 'Ag-110 Production')
plt.plot(xx, prod108, 'c', label = 'Ag-108 Production')
plt.plot(shiftextxData, extyFit110, 'g', label = 'Ag-110 Decay')
plt.plot(shiftextxData, extyFit108, 'm', label = 'Ag-108 Decay')
plt.plot(xx, prodtot, 'k', label = 'Total Production')
plt.plot(shiftextxData, extyFit, 'y', label = 'Total Decay')
plt.title('Production and Decay Curves for Ag-110 and Ag-108')
plt.xlabel('Time (s)')
plt.ylabel('Count Within 10 Second Interval')
plt.legend()
plt.savefig('totalcurve.png', dpi=300, bbox_inches='tight')
plt.close()

def ExpFitDiffEvolFixedbParam(x, y, b):
    """
    Fits a double-exponential decay Ae^-at + Be^-bt by scipy.optimize.differential_evolution with
    a fixed a parameter.

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    chi_sq : float
        chi squared as calculated by the sum of squared residuals
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    bounds = [[min(x), max(x)]]*2 + [[0, 1000]]*2

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum(((y - np.dot(fi, np.exp(-np.outer([b, 1./(taui[1])], x))))**2.)/(y))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    taui, fi = np.split(s, 2)
    return fi, [b, 1./(taui[1])], chi_sq

optChi_Sq = 42*red_chi_sq
posChi_Sq = 42*red_chi_sq
negChi_Sq = 42*red_chi_sq
optb = .00492
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    optb += 0.00001
    a, b, posChi_Sq = ExpFitDiffEvolFixedbParam(xData, yData, optb)
    a, b, negChi_Sq = ExpFitDiffEvolFixedbParam(xData, yData, (2*.00492-optb))

def ExpFitDiffEvolFixedaParam(x, y, a):
    """
    Fits a double-exponential decay Ae^-at + Be^-bt by scipy.optimize.differential_evolution with
    a fixed a parameter.

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    chi_sq : float
        chi squared as calculated by the sum of squared residuals
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.

    x = np.asarray(x)
    y = np.asarray(y)
"""    
    bounds = [[min(x), max(x)]]*2 + [[0, 1000]]*2

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum(((y - np.dot(fi, np.exp(-np.outer([a, 1./(taui[1])], x))))**2.)/(y))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    taui, fi = np.split(s, 2)
    return fi, [a, 1./(taui[1])], chi_sq

optChi_Sq = 42*red_chi_sq
posChi_Sq = 42*red_chi_sq
negChi_Sq = 42*red_chi_sq
opta = .0327
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    opta += 0.0001
    a, b, posChi_Sq = ExpFitDiffEvolFixedaParam(xData, yData, opta)
    a, b, negChi_Sq = ExpFitDiffEvolFixedaParam(xData, yData, (2*.0327-opta))

def ExpFitDiffEvolFixedBParam(x, y, B):
    """
    Fits a double-exponential decay Ae^-at + Be^-bt by scipy.optimize.differential_evolution with
    a fixed B parameter.

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    chi_sq : float
        chi squared as calculated by the sum of squared residuals
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.
"""

    x = np.asarray(x)
    y = np.asarray(y)
    
    bounds = [[min(x), max(x)]]*2 + [[0, 1000]]*2

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum(((y - np.dot([fi[0], B], np.exp(-np.outer(1./taui, x))))**2.)/(y))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    taui, fi = np.split(s, 2)
    return [fi[0], B], 1./taui, chi_sq

optChi_Sq = 42*red_chi_sq
posChi_Sq = 42*red_chi_sq
negChi_Sq = 42*red_chi_sq
optB = 30.2
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    optB += 0.1
    a, b, posChi_Sq = ExpFitDiffEvolFixedBParam(xData, yData, optB)
    a, b, negChi_Sq = ExpFitDiffEvolFixedBParam(xData, yData, (60.4-optB))

def ExpFitDiffEvolFixedAParam(x, y, A):
    """
    Fits a double-exponential decay Ae^-at + Be^-bt by scipy.optimize.differential_evolution with
    a fixed A parameter.

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    Returns
    -------
    a : (N) array
        solution prefactors in order a1, a2, ..., aN.
    b : (N) array
        solution exponents (positive) in order b1, b2, ..., bN.
    chi_sq : float
        chi squared as calculated by the sum of squared residuals
    Notes
    -----
    .. versionadded:: 1.0.0
    No notes yet.
"""
    x = np.asarray(x)
    y = np.asarray(y)
    
    bounds = [[min(x), max(x)]]*2 + [[0, 1000]]*2

    def objective(s):
        taui, fi = np.split(s, 2)
        return np.sum(((y - np.dot([A, fi[1]], np.exp(-np.outer(1./taui, x))))**2.)/(y))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    taui, fi = np.split(s, 2)
    return [A, fi[1]], 1./taui, chi_sq

optChi_Sq = 42*red_chi_sq
posChi_Sq = 42*red_chi_sq
negChi_Sq = 42*red_chi_sq
optA = 744
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    optA += 1
    a, b, posChi_Sq = ExpFitDiffEvolFixedAParam(xData, yData, optA)
    a, b, negChi_Sq = ExpFitDiffEvolFixedAParam(xData, yData, (2*744-optA))
print(optA-744)
print(optB-30.2)
print(opta-.0327)
print(optb-.00492)
