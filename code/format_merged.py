#!/usr/bin/env python

import pandas as pd

df = pd.read_csv('results/merged_interactions.tsv', sep='\t', index_col=[0,1,3])

ndf = df.unstack().fillna(0).droplevel(0, axis=1)

ndf.to_csv('results/format_merged_interactions.tsv', sep='\t')
