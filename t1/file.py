#!/usr/bin/python3
import numpy as np
import pandas as pd
import scipy.stats as stats

def MyNPV(Q,P,V,F,A,T,r,n,S,I):
    res = 0.0
    for t in range(int(n)):
        res += ((Q*(P-V)-F-A)*(1-T) + A) / ((1+r)**t)
    return res + S/((1+r)**n) - I

#def R(Q,P,V,F,A):
#    return (Q*(P-V))-F-A

def Rcp(df):
    res = 0.0
    for index, x in df.iterrows():
        res += (x['Q'] * (x['P'] - x['V']) - x['F'] - x['A']) * x['pr']
    return res

def DR(df):
    res = 0.0
    for index, x in df.iterrows():
        res += ((x['Q'] * (x['P'] - x['V']) - x['F'] - x['A']) ** 2) * x['pr']

    return res - Rcp(df)**2

def Z(df, val=5000):
    return (Rcp(df) - val) / np.sqrt(DR(df))


N = 10

Xdf = pd.read_csv('X.csv', delimiter=',')
Ydf = pd.read_csv('Y.csv', delimiter=',')

print('1) Чистая приведенная стоимость инвестиций')

for index, x in Xdf.iterrows():
    npv = MyNPV(x['Q'], x['P'], x['V'], x['F'], x['A'],
                x['T'], x['r'], x['n'], x['S'], x['I'])
    print(f'X prob: {x["pr"]}, npv: {round(npv, 2)}')
print()
for index, y in Ydf.iterrows():
    npv = MyNPV(y['Q'], y['P'], y['V'], y['F'], y['A'],
                y['T'], y['r'], y['n'], y['S'], y['I'] / N)
    print(f'Y prob: {y["pr"]}, npv: {round(npv, 2)}')

print('2) Средняя чистая приведенная стоимость инвестиций')
Xnpv = 0.0
for index, x in Xdf.iterrows():
    npv = MyNPV(x['Q'], x['P'], x['V'], x['F'], x['A'],
                x['T'], x['r'], x['n'], x['S'], x['I'])
    Xnpv += npv * x['pr']

Ynpv = 0.0
for index, y in Ydf.iterrows():
    npv = MyNPV(y['Q'], y['P'], y['V'], y['F'], y['A'],
                y['T'], y['r'], y['n'], y['S'], y['I'] / N)
    Ynpv += npv * y['pr']

print(f'Xnpv: {round(Xnpv, 2)} Ynpv: {round(Ynpv, 2)}')

print('3) Чувствительность')

Xs = Xdf.iloc[[2]]
Xq = float(Xs['Q'])
pnpv = 0.0
while True:
    npv = MyNPV(Xq, float(Xs['P']), float(Xs['V']), float(Xs['F']), float(Xs['A']),
                float(Xs['T']), float(Xs['r']), float(Xs['n']), float(Xs['S']), float(Xs['I']))
    #print(npv, Xq)
    if npv < 0:
        Xq += 5000
        break
    Xq -= 5000
    pnpv = npv

print(f'Чувствительность проекта X по объему выпуска: {Xq}, npv: {round(pnpv, 2)}')

Ys = Ydf.iloc[[2]]
Yq = float(Ys['Q'])
pnpv = 0.0
while True:
    npv = MyNPV(Yq, float(Ys['P']), float(Ys['V']), float(Ys['F']), float(Ys['A']),
                float(Ys['T']), float(Ys['r']), float(Ys['n']), float(Ys['S']), float(Ys['I'] / N))
    #print(npv, Xq)
    if npv < 0:
        Yq += 1000
        break
    Yq -= 1000
    pnpv = npv

print(f'Чувствительность проекта Y по объему выпуска: {Yq}, npv: {round(pnpv, 2)}')

Xp = float(Xs['P'])
pnpv = 0.0
while True:
    npv = MyNPV(float(Xs['P']), Xp, float(Xs['V']), float(Xs['F']), float(Xs['A']),
                float(Xs['T']), float(Xs['r']), float(Xs['n']), float(Xs['S']), float(Xs['I']))
    if npv < 0:
        Xp += 500
        break
    Xp -= 500
    pnpv = npv

print(f'Чувствительность проекта X по цене за штуку: {Xp}, npv: {round(pnpv, 2)}')

Yp = float(Ys['P'])
pnpv = 0.0
while True:
    npv = MyNPV(float(Ys['P']), Yp, float(Ys['V']), float(Ys['F']), float(Ys['A']),
                float(Ys['T']), float(Ys['r']), float(Ys['n']), float(Ys['S']), float(Ys['I']) / N)
    if npv < 0:
        Yp += 5000
        break
    Yp -= 5000
    pnpv = npv

print(f'Чувствительность проекта Y по цене за штуку: {Yp}, npv: {round(pnpv, 2)}')

print('4) Доходность')
print(f'Вероятность доходности <5000 для проекта X {stats.norm.pdf(Z(Xdf))}')
print(f'Вероятность доходности <5000 для проекта Y {stats.norm.pdf(Z(Ydf))}')

print('5) Доходность')
npv = -1.0
years = 0
while npv < 0:
    years+=1
    npv = MyNPV(float(Xs['Q']), float(Xs['P']), float(Xs['V']), float(Xs['F']), float(Xs['A']),
                float(Xs['T']), float(Xs['r']), years, float(Xs['S']), float(Xs['I']))

print(f'Проект X окупается через {years} лет,npv: {npv}')

npv = -1.0
years = 0
while npv < 0:
    years+=1
    npv = MyNPV(float(Ys['Q']), float(Ys['P']), float(Ys['V']), float(Ys['F']), float(Ys['A']),
                float(Ys['T']), float(Ys['r']), years, float(Ys['S']), float(Ys['I']) / N)

print(f'Проект Y окупается через {years} лет,npv: {npv}')
