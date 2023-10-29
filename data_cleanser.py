import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv("./TD_HOSPITAL_TRAIN.csv")

whitelist = ['death', 'cancer', 'diabetes', 'disability', 'dnr', 'primary', 'race', 'sex']

# Sort keys in alphabetical order, then reorganize the dataframe.
keys = df.keys()
keys = sorted(keys)
df = df[keys]

# The following are dictionaries for mapping mortality correlation.
dnr_dict = {'no dnr': 1,
            'dnr before sadm': 2,
            'dnr after sadm': 3
    }

primary_dict = {'Cirrhosis': 1,
                'Colon Cancer': 2,
                'ARF/MOSF w/Sepsis': 3,
                'COPD': 4,
                'Lung Cancer': 5,
                'Coma': 6,
                'CHF': 7,
                'MOSF w/Malig': 8
}

cancer_dict = {'yes' : 1,
               'no' : 2,
               'metastatic' : 3
}

# Does this even matter?
race_dict = {'white': 2,
             'black': 1.5,
             'hispanic': 1
}

disability_dict = {'<2 mo. follow-up' : 1,
                   'no(M2 and SIP pres)' : 2,
                    'SIP>=30' : 3,
                    'adl>=4 (>=5 if sur)' : 4,
                    'Coma or Intub' : 5
}

sex_dict = {'M' : 1,
            'F' : 2
}

extraprimary_dict = {'COPD/CHF/Cirrhosis' : 1,
                     'Cancer' : 2,
                     'ARF/MOSF' : 3,
                     'Coma' : 4
}

# Changes all values in column to a certain value.
def data_tester(df):
    for key in df.keys():
        if (df[key] == "COPD").any():
            df[key] = df[key].apply(lambda x: 1 if x == "COPD" else 0)
    
    df.to_csv("./TD_HOSPITAL_TRAIN_SCRAP.csv", index=False)

# Deletes certain parts of the data.
def delete_data(df):
    # 'bloodchem5' and 'psych2' were removed due to low varience. The rest was due to unnecessary data.
    to_drop = ['administratorcost', 'cost', 'dose', 'education', 'income', 'information', 'pdeath', 'sleep', 'timeknown', 'totalcost', 'psych3', "bloodchem5", "psych2"]

    # Check out the bar graph for these.
    to_drop += ['cancer', 'bloodchem3', 'bloodchem4', 'temperature']

    # Get rid of this if necessary.
    maybe_drop = ['pain', 'confidence']
    # df.drop(columns=maybe_drop, inplace=True)

    df.drop(columns=to_drop, inplace=True)

def alter_dataset(df):
    # Edits 'sex' column.
    for i in range(len(df['sex'])):
        if df.loc[i, 'sex'] == 'female' or df.loc[i, 'sex'] == 'Female':
            df.loc[i, 'sex'] = 'F'
        elif df.loc[i, 'sex'] == 'male' or df.loc[i, 'sex'] == 'Male':
            df.loc[i, 'sex'] = 'M'
        elif df.loc[i, 'sex'] == '1':
            df.loc[i, 'sex'] = np.NaN

alter_dataset(df)

# Replaces the outliers with NaN values in given column name.
def replace_range(df, column_name, max_range, min_range, value):
    df.loc[df[column_name] > max_range, column_name] = value
    df.loc[df[column_name] < min_range, column_name] = value

# Remove data that is not within the range of the given column name.
def trim_range(df, column_name, max_range, min_range):
    df = df[df[column_name] < max_range]
    df = df[df[column_name] > min_range]

# Takes in the dictionary above and formats it into a bar graph.
def death_rate_visualizer(df, dict, x):
    length = len(dict)

    mortality = [0] * length
    total_mortality = [0] * length
    death_rate = [0] * length

    death_index = 0

    row_count = df.shape[0]

    for i in range(row_count):
        primary_disease = df.loc[i, x]
        death_index = df.loc[i, 'death']

        if primary_disease in dict:
            total_mortality[dict[primary_disease] - 1] += 1 

            if death_index == 1:
                mortality[dict[primary_disease] - 1] += 1 

    for i in range(len(mortality)):
        death_rate[i] = (mortality[i] / total_mortality[i]) * 10

    print(death_rate)

    plt.xticks(fontsize=6)
    plt.bar(dict.keys(), death_rate)

    for i in range(len(death_rate)):
        death_rate[i] = round(death_rate[i], 1)

    plt.title(f"Mortality rate of {x}")
    plt.ylabel("Mortality rate")
    plt.show()

    return death_rate

# Stores and prints the amount of rows and data in the .csv file.
def column_and_row(df):
    dimensions = np.shape(df)
    rows, columns = dimensions
    print(rows)
    print(columns)

# Finds the variance of all numerical columns and removes if exceeds 0.5 after being put into tanh(x) function.
def variance_remover(df):
    tanh_list = []
    index_list = []
    non_tanh_list = []

    for i in range(len(df.columns)):
        try:
            if df.columns[i] in whitelist or '_' in df.columns[i]:
                tanh_list.append(1)
                continue

            variance_without_nan = np.nanvar(df[df.columns[i]])
            tanh_list.append(np.tanh(variance_without_nan))
            non_tanh_list.append(variance_without_nan)
            index_list.append(i)
        except:
            continue

    numDeleted = 0
    for i in range(len(index_list)):
        if tanh_list[i] < 0.5:
            # print(df.columns[i - numDeleted])
            df.drop(columns=[df.columns[i - numDeleted]], inplace=True)
            numDeleted += 1

# Maps each column against all of the other columns in order to find higher correlation.
def r_visualizer(df):
    for col1 in df.columns:
        if df[col1].dtype == 'object':
            continue

        r_list = []
        x_list = []
        for column in df.columns:
            try:
                if column != col1:
                    temp_list = [[], []]
                    for i in range(df.shape[0]):
                        if pd.notnull(df.loc[i, col1]) and pd.notnull(df.loc[i, column]):
                            temp_list[0].append(df.loc[i, col1])
                            temp_list[1].append(df.loc[i, column])
                        else:
                            continue

                    if len(temp_list[0]) > 1 and len(temp_list[1]) > 1:  # Checking if there are enough valid values for calculation
                        if abs(r2_score(temp_list[0], temp_list[1])) <= 10:
                            r_list.append(r2_score(temp_list[0], temp_list[1]))
                            x_list.append(column)
                    else:
                        r_list.append(np.nan)  # Append NaN for cases with insufficient valid values for calculation
                        x_list.append(column)
                else:
                    continue
            except:
                continue
        
        print(x_list)
        print(r_list)
        plt.xticks(rotation=45, fontsize=7)
        plt.title("R^2 values of inputted variable vs. the rest of the numeric data.")
        plt.tight_layout()
        plt.bar(x_list, r_list)
        plt.savefig(f'R^2 {col1}')
        plt.close()

# Changes the dataset to how we want it for training.
# alter_dataset(df)
# delete_data(df)

# Prints out all of the data and puts the graphs in the corresponding folder.
# print(r_visualizer(df))

weightings = {
    'primary': {
        'Cirrhosis': 6.50,
        'Colon Cancer': 8.40,
        'ARF/MOSF w/Sepsis': 5.80,
        'COPD': 5.80,
        'Lung Cancer': 8.90,
        'Coma': 8.00,
        'CHF': 6.10,
        'MOSF w/Malig': 8.90
    },
    'dnr': {
        'no dnr': 5.5254,
        'dnr before sadm': 8.54,
        'dnr after sadm': 9.02,
    },
    'disability': {'<2 mo. follow-up' : 9.78,
                'no(M2 and SIP pres)' : 4.7,
                'SIP>=30' : 5.39,
                'adl>=4 (>=5 if sur)' : 5.95,
                'Coma or Intub' : 7.037
                },
    'cancer': {
        'metastatic': 8.756,
        'no': 5.96,
        'yes': 7.53
    },
    'extraprimary': {
        'COPD/CHF/Cirrhosis' : 6.122,
                    'Cancer' : 8.696,
                    'ARF/MOSF' : 6.2826,
                    'Coma' : 8.06
    }
}

for key in weightings:
    for subkey in weightings[key]:
        df.loc[df[key] == subkey, key] = weightings[key].get(subkey, 0)

df.to_csv("./TD_HOSPITAL_TRAIN_PARSED.csv", index=False)