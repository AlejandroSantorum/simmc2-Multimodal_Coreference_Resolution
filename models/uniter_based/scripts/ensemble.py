import json
import pandas as pd
from glob import glob
import os
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import optuna
import argparse
import matplotlib.pyplot as plt

def read_data(mode='valid'):
    # print('mode', mode)
    paths = glob('../output/logit_ensemble/*.json')
    valid_paths = [x for x in paths if 'learnableKB' not in x and 'dev.json' in x]
    test_paths = [x[:-5]+'test.json' for x in valid_paths]
    vps, tps = [], []
    for tp in test_paths:
        if os.path.exists(tp):
            vps.append(tp[:-9]+'.json')
            tps.append(tp)
    valid_paths = vps
    test_paths = tps
    del vps, tps
    assert all([os.path.exists(x) for x in test_paths])
    print("number of models:", len(valid_paths))
    names = [vp.split('/')[-1].split('.')[0] for vp in valid_paths]
    # print(test_paths[0])
    x = defaultdict(list)
    y = {}
    ps = valid_paths if mode == 'valid' else test_paths
    for p in ps:
        # print(p)
        with open(p, 'r') as f:
            d = json.load(f)
        for k in d:
            for kk, tuples in d[k].items():
                outk = k + '-' + kk
                objy = []
                objx = []
                for i, obj in enumerate(tuples):
                    objx.append(obj[0])
                    objy.append(obj[1])
                x[outk].append(objx)
                if outk not in y:
                    y[outk] = objy
                else:
                    assert y[outk] == objy, f'{outk} {i}'
    X = []
    yy = []
    for k, v in x.items():
        # v.shape = (nmodels, nobjs)
        labels = y[k]
        yy.extend(labels)
        v = np.array(v).T
        # print(v.shape)
        X.append(v)
        
    X = np.concatenate(X, axis=0)
    yy = np.array(yy)
    # print(X.shape)
    # print(y.shape)
    # print("number of turns:", len(y))
    # print("number of objects:", len(x[outk]))
    # tempx = []
    # tempy = []
    # for k in x:
    #     tempx.append(x[k])
    #     tempy.append(y[k])
    # xlengs = [len(xx) for xx in x]
    # ylengs = [len(yy) for yy in y]
    # x = np.array(tempx)
    # y = np.array(tempy)
    # del tempx, tempy
    # print("x shape:", x.shape)
    # print("y shape:", y.shape)
    # print(X[0], y[0])
    return X, yy, x, y, names

def loss_func(weights):
    avg_logit = np.average(X, axis=1, weights=weights)
    predictions = (avg_logit > 0).astype(int)
    return -f1_score(y, predictions)

def objective(trial):
    ws = np.zeros(nmodels)
    for i in range(nmodels):
        ws[i] = trial.suggest_float(f'w{i}', -1, 1)
    return loss_func(ws)

if __name__ == '__main__':
    X, y, Xd, yd, names = read_data(mode='valid')
    nmodels = X.shape[1]
    starting_values = [1/nmodels]*nmodels
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['optuna', 'nelder-mead'], default='optuna',
                        type=str)
    args = parser.parse_args()
    print("optimizing method:", args.method)
    # cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    # bounds = [(0,1)]*nmodels
    # res = minimize(loss_func, starting_values, method='Nelder-Mead', bounds=bounds, constraints=cons)
    if args.method == 'nelder-mead':
        res = minimize(loss_func, starting_values, method='Nelder-Mead')
        print('finding weights on dev')
        print('ensembled dev f1: {best_score}'.format(best_score=-res['fun']))
        print('best dev f1 score: {weights}'.format(weights=res['x']))
        weights = res['x']
    else:
        print('finding weights on dev')
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        weights = []
        for i in range(nmodels):
            weights.append(study.best_params[f'w{i}'])
        print('ensembled dev f1: {best_score}'.format(best_score=-study.best_trial.value))
        print('best dev f1 score: {weights}'.format(weights=weights))
    print('predicting on devtest')
    X, y, Xd, yd, names = read_data(mode='test')
    ens_devtest_score =  -loss_func(weights)
    print('ensembled devtest f1', ens_devtest_score)
    print('single model scores on devtest:')
    single_scores = []
    for i in range(nmodels):
        weights = np.zeros(nmodels)
        weights[i] = 1
        single_scores.append(-loss_func(weights))
        print(names[i], single_scores[-1])
    bl = plt.bar(names+['ensembled'], single_scores+[ens_devtest_score])
    ax = plt.gca()
    ax.set_ylim([min(single_scores)//0.1*0.1, (ens_devtest_score//0.1+1)*0.1])
    bl[-1].set_color('r')
    plt.xlabel('model')
    plt.ylabel('f1@devtest')
    plt.xticks(rotation=90)
    # ax.subplots_adjust(bottom=0.5)
    plt.show()
    print('saving blended devtest')
    blend_output = defaultdict(lambda: defaultdict(list))
    for k, v in Xd.items():
        dialog_idx = k.split('-')[0]
        round_idx = k.split('-')[1]
        # v.shape = (nmodels, nobjs)
        v = np.array(v)
        v_ens = np.average(v, axis=0, weights=weights)
        ys = yd[k]
        for i, y in enumerate(ys):
            blend_output[dialog_idx][round_idx].append([v_ens[i], y])
        # print(v)
        # print(len(v))
    with open('../output/logit_ensemble/blended_devtest.json', 'w') as f:
        json.dump(blend_output, f, indent=4)
    print('done!')