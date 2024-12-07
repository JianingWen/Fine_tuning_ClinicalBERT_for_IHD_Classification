import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_notes = pd.read_csv("Dataset/notes.csv")
df_notes = df_notes.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])

df_icd9 = pd.read_csv('Dataset/icd9.csv')
df_icd9['ICD9_CODE'] = df_icd9['ICD9_CODE'].fillna('Unknown')

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
                'ID': df_notes.SUBJECT_ID.iloc[i]
            })
        if len(x) % 318 > 10:
            rows.append({
                'TEXT': ' '.join(x[-(len(x) % 318):]),
                'Label': df_notes.HAS_CAD.iloc[i],
                'ID': df_notes.SUBJECT_ID.iloc[i]
            })

    want = pd.DataFrame(rows)
    return want

df_notes = preprocessing(df_notes)
df_notes['Label'] = df_notes['Label'].astype(int)

# Split train, val, and test files by patients
subject_ids = df_notes['ID'].unique()

train_ids = np.random.choice(subject_ids, size=2000, replace=False)
val_ids = np.random.choice([id for id in subject_ids if id not in train_ids], size=1000, replace=False)
test_ids = np.random.choice([id for id in subject_ids if id not in train_ids and id not in val_ids], size=1000, replace=False)

df_train = df_notes[df_notes['ID'].isin(train_ids)]
df_val = df_notes[df_notes['ID'].isin(val_ids)]
df_test = df_notes[df_notes['ID'].isin(test_ids)]

df_train.to_csv("train_processed.csv", index=False)
df_val.to_csv("val_processed.csv", index=False)
df_test.to_csv("test_processed.csv", index=False)
