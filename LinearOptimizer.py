import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

xData = [7, 7.25, 7.5, 7.75, 8, 8.25, 8.5]
yData = [8.24, 8.53, 8.87, 9.18, 9.48, 9.98, 10.32]
unc = [.05, .05, .06, .03, .01, .03, .03]

def LinearFit(x, y, u):
    """
    Fits a linear model by scipy.optimize.differential_evolution

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    u : (n) array_like
        1-dimensional list of uncertainties for y-values.
    Returns
    -------
    a, b : Optimal parameters for linear model a + bx
    red_chi_sq : float
        reduced chi squared as calculated by the sum of squared residuals
    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-1000, 1000]] + [[-1000, 1000]]

    def objective(s):
        a, b = np.split(s, 2)
        return np.sum(((y - b*x-a)**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    a, b = np.split(s, 2)
    return a, b, red_chi_sq

a, b, red_chi_sq = LinearFit(xData, yData, unc)
print(a)
print(b)
print(red_chi_sq)

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yData, yerr=unc, fmt='ro')
plt.plot(xData, a+b*xData, 'b', label = '1.41x-1.75')
plt.title('Angular Frequency vs Voltage')
plt.xlabel('Voltage (V)')
plt.ylabel('Angular Frequency (rad/sec)')
plt.legend()
plt.savefig('freq.png', dpi=300, bbox_inches='tight')
plt.close()

height = [3.1, 4.8, 4.1, 3.5, 2.4]
upper = [9.55, 9.97, 9.74, 9.64, 9.33]
lower = [8.36, 8.27, 8.30, 8.32, 8.42]
upunc = [.34, .35, .35, .34, .34]
lowunc = [.32, .32, .32, .32, .32]

aUp, bUp, red_chi_sqUp = LinearFit(height, upper, upunc)
print(aUp)
print(bUp)
print(red_chi_sqUp)

aLow, bLow, red_chi_sqLow = LinearFit(height, lower, lowunc)
print(aLow)
print(bLow)
print(red_chi_sqLow)

#plot of best fit vs data
fig = plt.figure()
plt.errorbar(height, upper, yerr=upunc, fmt='ro')
plt.errorbar(height, lower, yerr=lowunc, fmt='bo')
plt.plot(height, aUp+bUp*height, 'r', label = 'Upper Bound Fit: .255x+8.73')
plt.plot(height, aLow+bLow*height, 'b', label = 'Lower Bound Fit: -.0616x+8.55')
plt.title('Upper and Lower Bounds on Unstable Region')
plt.xlabel('Height Modulation Amplitude (cm)')
plt.ylabel('Angular Frequency (rad/sec)')
plt.legend()
plt.savefig('bounds.png', dpi=300, bbox_inches='tight')
plt.close()

xx = np.linspace(2.4, 4.8, num=50)
theoryUp = []
theoryLow = []
for i in xx:
    theoryUp.append(0.193*i+9.12)
    theoryLow.append((-0.193)*i+9.12)

#plot of theoretical fit vs data
fig = plt.figure()
plt.errorbar(height, upper, yerr=upunc, fmt='ro')
plt.errorbar(height, lower, yerr=lowunc, fmt='bo')
plt.plot(xx, theoryUp, 'r', label = 'Theoretical Upper Bound: .193x+9.12')
plt.plot(xx, theoryLow, 'b', label = 'Theoretical Lower Bound: -.193x+9.12')
plt.title('Theoretical Upper and Lower Bounds on Unstable Region')
plt.xlabel('Height Modulation Amplitude (cm)')
plt.ylabel('Angular Frequency (rad/sec)')
plt.legend()
plt.savefig('theory.png', dpi=300, bbox_inches='tight')
plt.close()

theoryUpPt = [9.72, 10.05, 9.91, 9.79, 9.58]
theoryLowPt = [8.52, 8.19, 8.33, 8.44, 8.65]
UpResid = np.subtract(upper, theoryUpPt)
LowResid = np.subtract(lower, theoryLowPt)

#Upper theoretical residual
fig = plt.figure()
plt.errorbar(height, UpResid, yerr=upunc, fmt='ro')
plt.title('Residual Plot of Theoretical Upper Bounds')
plt.xlabel('Height Modulation Amplitude (cm)')
plt.ylabel('Angular Frequency (rad/sec)')
plt.savefig('upperresiduals.png', dpi=300, bbox_inches='tight')
plt.close()

#Lower theoretical residual
fig = plt.figure()
plt.errorbar(height, LowResid, yerr=lowunc, fmt='bo')
plt.title('Residual Plot of Theoretical Lower Bounds')
plt.xlabel('Height Modulation Amplitude (cm)')
plt.ylabel('Angular Frequency (rad/sec)')
plt.savefig('lowerresiduals.png', dpi=300, bbox_inches='tight')
plt.close()

logAngle = [2.3, 2.08, 1.79, 1.39, 0.69]
time = [0, 32.16, 74.06, 133.65, 220.74]
timeunc = [0.01, 0.83, 1.12, 3.20, 6.62]

aDamp, bDamp, red_chi_sqDamp = LinearFit(logAngle, time, timeunc)
print(aDamp)
print(bDamp)
print(red_chi_sqDamp)

#plot of fit vs data
fig = plt.figure()
plt.errorbar(logAngle, time, yerr=timeunc, fmt='ro')
plt.plot(logAngle, aDamp+bDamp*logAngle, 'b', label = '-144.55x + 332.46')
plt.title('Log of Amplitude vs Time for Damped Pendulum')
plt.ylabel('Time (s)')
plt.xlabel('Log of Angle (Degrees)')
plt.legend()
plt.savefig('damping.png', dpi=300, bbox_inches='tight')
plt.close()
