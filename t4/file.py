#!/usr/bin/python3
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def var_gaussian(r, level=0.01, modified=False):
    """
    Returns the Parametric Gauuian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = stats.norm.ppf(0.01)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = stats.skew(r)
        k = stats.kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

df = pd.read_excel('data.xlsx')
#df = pd.read_csv('data.csv', delimiter=',')
data = np.array(df['curs'])

print('1)')
print(df)


print('\n2)')
print(f'len: {len(data)}')
print(f'mean: {data.mean()}')
print(f'var: {data.var()}')

print('\n3)')
data20 = data.reshape(20, -1)
print(f'min: {data.min()} max: {data.max()}')

X = data.mean()
DX = data.var()
b = (np.sqrt(12*DX) + 2*X) * 0.5
a = 2*X - b
print(f'a: {a} b: {b}')

print('\n4)')

print('\n5)')

alpha = (2*DX) / (DX - X**2)
x0 = (alpha - 1) * X

if x0 < 0:
    x0 = data.min()

if alpha < 0:
    alpha = len(data) / np.sum([np.log(x) / x0 for x in data])

print(f'alpha: {alpha} x_0: {x0}')

def pareto(x):
    return alpha / (x**(alpha+1))

plt.hist(data)
#plt.show()
plt.savefig('pareto.png')
print(stats.kstest(data, pareto))


print('\n6)')

df['year'] = df['data'].dt.year
#print(df)
xtrms = []
for i in range(2002, 2022):
    xtrms.append(np.max(np.array(df.query(f'year == {i}')['curs'])))

xtrms.sort(reverse=True)

k = len(xtrms)
G = np.sum([np.log(xtrms[i]/xtrms[i+1]) for i in range(k-1)])
chi = 1/G
print(f'G: {G} chi: {chi}')

DXm = np.var(xtrms); Xm = np.mean(xtrms)

delta = np.sqrt(abs((DXm * chi) / (math.gamma(1 - 2*chi) - math.gamma(1 - chi)**2)))
mu = X - (delta * (math.gamma(1 - chi) - 1))

if chi > 0.5:
    delta = 1; mu = 0
print(f'delta: {delta} mu: {mu}')

pv = 1 - np.exp(-(1 + chi*((1.1*max(xtrms) - mu)/delta) ** (-1/chi)))
print(f'prob: {pv}')

print('\n7)')


print(f'30 летний уровень стоимости: {mu + delta*((-np.log(1-1/30))**-chi - 1)/chi}')

print('\n8)')

inv = 2e6
VAR = var_gaussian(data)
if VAR < 0:
    str = 'loss'
else:
    str = 'gain'

print(f'VaR: {VAR}%\ninvestement: {inv} {str}: {round(abs(inv*(VAR/100)),2)}')
