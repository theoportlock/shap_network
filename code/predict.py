#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imblearn.over_sampling import smote
from itertools import permutations
from pathlib import path
from sklearn.ensemble import randomforestclassifier, randomforestregressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import standardscaler
import argparse
import numpy as np
import os
import pandas as pd
import shap
import sys

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=true):
    output_path = f'results/{subject}.tsv' 
    os.makedirs(os.path.dirname(output_path), exist_ok=true)
    df.to_csv(output_path, sep='\t', index=index)

def predict(df, analysis, shap_val=false, shap_interact=false, n_iter=30):
    outputs = []
    aucrocs = []
    fpr_tpr = []
    maes = []
    r2s = []
    meanabsshaps = pd.dataframe()
    shap_interacts = pd.dataframe(index=pd.multiindex.from_tuples(permutations(df.columns, 2)))
    
    random_state = 1
    for random_state in range(n_iter):
        if analysis.lower() == 'classifier':
            model = randomforestclassifier(n_jobs=-1, random_state=random_state, bootstrap=false, max_depth=none, min_samples_leaf=8,min_samples_split=20,n_estimators=400)
            # updated from the hyperparameter tuning
        elif analysis.lower() == 'regressor':
            model = randomforestregressor(n_jobs=-1, random_state=random_state)
        else:
            raise valueerror("invalid analysis type. choose 'classifier' or 'regressor'.")
 
        x, y = df, df.index

        scaler = standardscaler()
        x_scaled = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=random_state, stratify=y)
        smoter = smote(random_state=random_state)
        x_train_upsample, y_train_upsample = smoter.fit_resample(x_train, y_train)

        model.fit(x_train_upsample, y_train_upsample)
        
        if analysis.lower() == 'classifier':
            y_prob = model.predict_proba(x_test)[:, 1]
            aucrocs.append(roc_auc_score(y_test, y_prob))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            aucrocdata = pd.dataframe({
                'fpr': fpr,
                'tpr': tpr,
                'random_state': random_state
            })
            fpr_tpr.append(aucrocdata)
        
        if analysis.lower() == 'regressor':
            y_pred = model.predict(x_test)
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))
        
        if shap_val:
            explainer = shap.treeexplainer(model)
            shap_values = explainer(x_scaled)
            meanabsshaps[random_state] = pd.series(
                    np.abs(shap_values.values[:,:,1]).mean(axis=0),
                    index=x.columns
            )
       
        if shap_interact:
            explainer = shap.treeexplainer(model)
            explainer = shap.explainer(model)
            inter_shaps_values = explainer.shap_interaction_values(x_scaled)
            sum_shap_interacts = pd.dataframe(
                data=inter_shaps_values[:,:,:,1].sum(0),
                columns=df.columns,
                index=df.columns)
            shap_interacts[random_state] = sum_shap_interacts.stack()

    fpr_tpr_df = pd.concat(fpr_tpr)
    save(pd.series(aucrocs).to_frame(f'{subject}'), f'{subject}aucrocs', index=false)
    save(meanabsshaps, f'{subject}meanabsshaps')
    save(shap_interacts,f'shap_interacts')
    save(fpr_tpr_df, f'{subject}fpr_tpr')

def parse_args(args):
    parser = argparse.argumentparser(
       prog='predict.py',
       description='random forest classifier/regressor with options'
    )
    parser.add_argument('analysis', type=str, help='regressor or classifier')
    parser.add_argument('subject', type=str, help='data name or full filepath')
    parser.add_argument('-n','--n_iter', type=int, help='number of iterations for bootstrapping', default=10)
    parser.add_argument('--shap_val', action='store_true', help='shap interpreted output')
    parser.add_argument('--shap_interact', action='store_true', help='shap interaction interpreted output')
    return parser.parse_args(args)

arguments = sys.argv[1:]
args = parse_args(arguments)

# check if the provided subject is a valid file path
if os.path.isfile(args.subject):
    subject = path(args.subject).stem
else:
    subject = args.subject

df = load(subject)
analysis = args.analysis
shap_val = args.shap_val
shap_interact = args.shap_interact
n_iter = args.n_iter

predict(df, args.analysis, shap_val=args.shap_val, shap_interact=args.shap_interact, n_iter=args.n_iter)
