#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys
#sys.path.append('../common/')
#from tex import tex
import scipy.stats
def t(alpha, gl):
    return scipy.stats.t.ppf(1-(alpha/2), gl)


def income(rts):
    r = np.zeros(len(rts)-1)
    for i in range(len(rts)-1):
        r[i] = (rts[i+1] - rts[i]) / rts[i]
    return r

df = pd.read_csv('candels1.csv', delimiter=';')
df1 = pd.read_csv('candels2.csv', delimiter=';')
df2 = pd.read_csv('candels3.csv', delimiter=';')
RTS = np.array(df[['close']]).T[0]
RTS1 = np.array(df1[['close']]).T[0]
RTS2 = np.array(df2[['close']]).T[0]

#1
print('1) Доходность')
R = income(RTS)
R1 = income(RTS1)
R2 = income(RTS2)
print(f'R: {R}\n R1: {R1}\n R2: {R2}')
plt.plot(R)
plt.plot(R1)
plt.plot(R2)
plt.legend(('R', 'R1', 'R2'))
plt.savefig('plot1.png')
plt.clf()
#2
print('\n2) Регрессия')
a, b = np.polyfit(R, R1, 1)
c, d = np.polyfit(R, R2, 1)
print(f'beta1: {a} beta2: {c}')
plt.plot([a * x + b for x in range(len(R))])
plt.plot([c * x + d for x in range(len(R))])
plt.legend((f'beta1:{round(a, 2)}', f'beta2:{round(c, 2)}'))
plt.savefig('plot2.png')
plt.clf()
#3
print('\n3) Оценки рисков')
residue1 = R1 - np.array([a * x + b for x in R])
residue2 = R2 - np.array([c * x + d for x in R])
risk1 = residue1.var()
risk2 = residue2.var()
risk = R.var()
print(f'risk: {risk} risk1: {risk1} risk2: {risk2}')
#4
print('\n4)')
#def delta(arr):
#    x1 = arr[0]; x2 = arr[1]
#    return

#5
print('\n5)')

#6
print('\n6) Значимость')
from sklearn.metrics import r2_score
r21 = r2_score(R1 ,[a * x + b for x in R])
r22 = r2_score(R2 ,[c * x + d for x in R])
print(f'r squared for r1: {r21}, root: {np.sqrt(r21)}')
print(f'r squared for r2: {r22}, root: {np.sqrt(r22)}')

#7
print('\n7) Гетероскедастичность')
sp1 = scipy.stats.spearmanr(R, R1)
t1 = t((1 - sp1[1]) / 2, 13 - 2 - 1)
sp2 = scipy.stats.spearmanr(R, R2)
t2 = t((1 - sp2[1]) / 2, 13 - 2 - 1)
print(f'sp1 {t1 < sp1[1]}')
print(f'sp2 {t2 < sp2[1]}')
#8
print('\n8) Дарбин-Уотсон')
from statsmodels.stats.stattools import durbin_watson
test1 = durbin_watson(residue1)
test2 = durbin_watson(residue1)
print(f'dw test result: {test1}, {test1 > 2}')
print(f'dw test result: {test2}, {test2 > 2}')
