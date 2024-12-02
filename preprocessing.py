import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_notes = pd.read_csv("Dataset/notes.csv")
df_notes = df_notes.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])

df_icd9 = pd.read_csv('Dataset/icd9.csv')

# Create HAS_CAD Label (has coronary artery disease)
cad_codes = ['410', '411', '412', '413', '414']
icd9_with_cad = df_icd9.loc[df_icd9['ICD9_CODE'].str.startswith(tuple(cad_codes)), 'SUBJECT_ID']
df_notes['HAS_CAD'] = df_notes['SUBJECT_ID'].isin(icd9_with_cad)

# Preprocess text - adapted from https://github.com/kexinhuang12345/clinicalBERT/blob/master/preprocess.py 
import re
from tqdm import tqdm

def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y

def preprocessing(df_notes):
    df_notes['TEXT'] = df_notes['TEXT'].fillna(' ')
    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\n', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\r', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].apply(str.strip)
    df_notes['TEXT'] = df_notes['TEXT'].str.lower()

    df_notes['TEXT'] = df_notes['TEXT'].apply(lambda x: preprocess1(x))

    # to get 318 words chunks 
    rows = []
    for i in tqdm(range(len(df_notes))):
        x = df_notes.TEXT.iloc[i].split()
        n = int(len(x) / 318)
        for j in range(n):
            rows.append({
                'TEXT': ' '.join(x[j * 318:(j + 1) * 318]),
                'Label': df_notes.HAS_CAD.iloc[i],
                'ID': df_notes.HADM_ID.iloc[i]
            })
        if len(x) % 318 > 10:
            rows.append({
                'TEXT': ' '.join(x[-(len(x) % 318):]),
                'Label': df_notes.HAS_CAD.iloc[i],
                'ID': df_notes.HADM_ID.iloc[i]
            })

    want = pd.DataFrame(rows)
    return want

df_notes = preprocessing(df_notes)
df_notes['Label'] = df_notes['Label'].astype(int)
# df_notes.to_csv("processed_notes.csv", index=False)
