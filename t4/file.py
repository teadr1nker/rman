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

h = (max(data) - min(data)) / 20
print(f'step h: {h}')
n = len(data)
data.sort()
#print(data)
intervals = []
count = 0
for i in range(20):
    interval = []
    while count < n and data[count] < data[0] + h*(i+1):
        interval.append(data[count])
        count += 1
    intervals.append(interval)

F = lambda x: (x - a)/(b - a)
p = lambda i: F(max(i)) - F(min(i))

chi2 = sum([
((len(i) - n * p(i)) ** 2) / n * p(i) for i in intervals
])
chi2tabl = 27.6
print(f'chi2: {chi2} chi2 crit: {chi2tabl}')
if chi2tabl > chi2:
    print('H0 верна')
else:
    print('H0 отвергается')


print('\n4)')

a = X; delta = np.sqrt(DX)
ranges = [
(a - 3*delta, a - 2*delta),
(a - 2*delta, a - delta),
(a - delta, a),
(a, a + delta),
(a + delta, a + 2*delta),
(a + 2*delta, a + 3*delta)
]

probs = [0.0217, 0.137, 0.34, 0.34, 0.34, 0.137, 0.0217]

intervals = []
count = 0
for m, M in ranges:
    interval = []
    while count < n and data[count] >= m and data[count] < M:
        interval.append(data[count])
        count += 1
    intervals.append(interval)

chi2 = 0
for x, i in enumerate(intervals):
    if len(i) > 0:
        chi2 += ((len(i) - n*probs[x])**2) / (n * probs[x])

chi2tabl = 7.8
print(f'chi2: {chi2} chi2 crit: {chi2tabl}')
if chi2tabl > chi2:
    print('H0 верна')
else:
    print('H0 отвергается')

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
