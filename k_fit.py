import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar

def mse(w1, y):
    return np.mean((w1 - y)**2)

def k1(T, c1, c2, n):
    return c1 * c2 * T ** (n) / (c2 + c1 * T ** (n))

def k2(T, c3, c4):
    return c3*np.exp(-1 * c4 / T)



def open_file(path):
    with open(path, 'r') as f:
        data_raw = f.readlines()
    data_raw = [d.replace(' ', '\t').replace('\n', '') for d in data_raw]
    data = [d.split('\t') for d in data_raw]
    data = pd.DataFrame(data[1:], columns=['T', 'k'])
    return data

def preprocess(data):
    for c in data.columns:
        data[c] = data[c].astype('float')
    data['T_log'] = np.log(data['T'])
    data['k_log'] = np.log(data['k'])
    return data

def ploynomal_fit(data):
    z = np.polyfit(data['T_log'], data['k_log'], 7)
    p = np.poly1d(z)
    return p

def interpolate(p, min, max):
    xp = np.linspace(min, max, 200)
    ap_df = pd.DataFrame({'x': xp})
    ap_df['y'] = [p(x) for x in xp]
    return ap_df

def find_bend_points(p):
    p3 = np.polyder(p, m=3)
    roots = np.roots(p3)
    return sorted([x for x in roots if (np.exp(x)>2) and (np.exp(x)<=70)])

def refine_palteue(interp_df, bent_pts):
    plateue = interp_df[(interp_df['x'] > bent_pts[0]) & (interp_df['x'] < bent_pts[1])]
    plateue['ae'] = plateue['y'].apply(lambda x: plateue['y'].mean() - x)
    var = np.var(plateue['ae'])
    plateue = plateue[plateue['ae'].abs() < 3*var]
    return plateue[['x', 'y']].apply(lambda x: np.exp(x))


def approximate(data):
    data = preprocess(data)
    polynom = ploynomal_fit(data)
    bent_pts = find_bend_points(polynom)
    interp_df_log = interpolate(polynom, data['T_log'].min(), data['T_log'].max())
    plateue = refine_palteue(interp_df_log, bent_pts)

    T_pl_start = plateue['x'].min()
    k_pl = np.float(plateue.loc[plateue['x'] == plateue['x'].min(), 'y'])
    T = np.exp(interp_df_log['x'])
    k = np.exp(interp_df_log['y'])

    bel_pl = T < T_pl_start
    T_bel_pl = T[bel_pl]
    k_bel_pl = k[bel_pl]
    T_ab_pl = T[~bel_pl]
    k_ab_pl = k[~bel_pl]


    def loss_k1(*params):
        c1 = params[0][0]
        c2 = params[0][1]
        n = params[0][2]
        k_fit = k1(T_bel_pl, c1, c2, n)
        return np.mean((k_fit - k_bel_pl) ** 2)

    k1_params = minimize(loss_k1, (0.5, 0.2, 1),
                         method='SLSQP',
                         tol=1e-10,
                         options={'maxiter': 10000}
                         )
    c1, c2, n = tuple(k1_params.x.tolist())

    def loss_k2(*params):
        c3 = params[0][0]
        c4 = params[0][1]
        k_fit = k2(T_ab_pl, c3, c4)
        return np.mean((k_fit - (k_ab_pl - k_pl)) ** 2)

    k2_params = minimize(loss_k2, (2, 15),
                         method='SLSQP',
                         tol=1e-20,
                         options={'maxiter': 1000},
                         args=(c1, c2, n)
                         )

    c3, c4 = tuple(k2_params.x.tolist())
    return c1, c2, n, c3, c4 ,T

if __name__ == "__main__":
     data = open_file('/home/quantum/Documents/iltpe/Cu10Zn2Sb4S13.dat')
     c1, c2, n, c3, c4, T = approximate(data)
     print('C1 {}, C2 {}, N {}, C3 {}, C4 {}'.format(c1, c2, n, c3, c4, T))

     k_fit = k1(T, c1, c2, n) + k2(T, c3, c4)

     plt.scatter(data['T'], data['k'], color='g')
     plt.plot(T, k_fit)
     plt.yscale('log')
     plt.xscale('log')