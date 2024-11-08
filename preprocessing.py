import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_notes = pd.read_csv("notes.csv")
df_notes = df_notes.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])

df_icd9 = pd.read_csv('icd9.csv')

# Create HAS_CAD Label (has coronary artery disease)
cad_codes = ['410', '411', '412', '413', '414']
icd9_with_cad = df_icd9.loc[df_icd9['ICD9_CODE'].str.startswith(tuple(cad_codes)), 'SUBJECT_ID']
df_notes['HAS_CAD'] = df_notes['SUBJECT_ID'].isin(icd9_with_cad)
