import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.regularizers import l2


def data_preprocessing(df: pd.DataFrame):
    # test_list = ['age', 'blood', 'bloodchem3', 'bloodchem4', 'bloodchem5', 'breathing', 'cancer', 'comorbidity', 'confidence', 'death', 'diabetes', 'disability', 'dnr', 'extraprimary', 'glucose', 'heart', 'information', 'meals', 'pain', 'primary', 'psych1', 'psych5', 'psych6', 'race']
    # col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    
    
    # col_to_keep = ['death', 'age', 'blood', 'reflex', 'diabetes', 'breathing', 'bloodchem1', 'bloodchem2', 'bloodchem3', 'bloodchem4','bloodchem6', 'glucose', 'heart', 'psych5', 'psych4', 'psych6', 'psych1', 'comorbidity', 'temperature', 'cancer', 'dnr', 'disability', 'primary', 'sex_F', 'sex_M', 'pdeath']

    
    # ORIGINAL LABEL (~44%)
        # labels = ['age', 'blood', 'reflex', 'diabetes', 'bloodchem1', 'bloodchem6', 'breathing', 'glucose', 'psych5', 'comorbidity', 'sleep', 'temperature', 'cancer_metastatic', 'cancer_no', 'cancer_yes', 'disability_<2 mo. follow-up', 'disability_Coma or Intub', 'disability_SIP>=30', 'disability_adl>=4 (>=5 if sur)', 'disability_no(M2 and SIP pres)']

    # col_to_keep = ['death', 'age', 'blood', 'bloodchem1', 'bloodchem2', 'bloodchem3',  'bloodchem6', 'bp', 'breathing', 'comorbidity',
    #    'confidence', 'diabetes', 'glucose', 'heart', 'information',
    #    'meals', 'pain', 'psych1', 'psych6',
    #    'reflex', 'sleep', 'temperature', 'urine',
    #    'cancer_metastatic', 'cancer_no', 'cancer_yes',
    #    'disability_<2 mo. follow-up', 'disability_Coma or Intub',
    #    'disability_SIP>=30', 'disability_adl>=4 (>=5 if sur)',
    #    'disability_no(M2 and SIP pres)', 'dnr_dnr after sadm',
    #    'dnr_dnr before sadm', 'dnr_no dnr', 'primary_ARF/MOSF w/Sepsis', 'primary_CHF',
    #    'primary_COPD', 'primary_Cirrhosis', 'primary_Colon Cancer',
    #    'primary_Coma', 'primary_Lung Cancer', 'primary_MOSF w/Malig',
    #    'race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white',
    #    'sex_F', 'sex_M']
    
    # keys for tht_out:
    # col_to_keep = ['death', 'age', 'blood', 'bloodchem1', 'bloodchem2', 'bloodchem3',  'bloodchem6', 'bp', 'breathing', 'comorbidity',
    #    'confidence', 'diabetes', 'heart',
    #     'pain', 'psych1', 'psych6',
    #    'reflex', 'sleep', 'temperature',
    #    'cancer_metastatic', 'cancer_no', 'cancer_yes',
    #    'disability', 'dnr_dnr after sadm',
    #    'dnr_dnr before sadm', 'dnr_no dnr', 'primary',
    #    'race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white',
    #    'sex_F', 'sex_M']

    # keys for final:
    # col_to_keep = ['death', 'age', 'blood', 'bloodchem1', 'bloodchem4',  'breathing', 'comorbidity', 'diabetes', 'glucose', 'heart', 'meals', 'pain', 'psych1', 'psych5', 'psych6', 'cancer_metastatic', 'cancer_no', 'cancer_yes', 'disability_<2 mo. follow-up', 'disability_Coma or Intub', 'disability_SIP>=30', 'disability_adl>=4 (>=5 if sur)', 'disability_no(M2 and SIP pres)', 'dnr_dnr after sadm', 'dnr_dnr before sadm', 'dnr_no dnr', 'extraprimary_ARF/MOSF', 'extraprimary_COPD/CHF/Cirrhosis', 'extraprimary_Cancer', 'extraprimary_Coma', 'primary_ARF/MOSF w/Sepsis', 'primary_CHF', 'primary_COPD', 'primary_Cirrhosis', 'primary_Colon Cancer', 'primary_Coma', 'primary_Lung Cancer', 'primary_MOSF w/Malig', 'race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white', 'sex_F', 'sex_M']

    col_to_keep = ['death', 'age', 'blood', 'bloodchem1', 'bloodchem2', 'bloodchem6',  'bp', 'breathing', 'comorbidity',
       'confidence', 'diabetes', 'glucose', 'heart', 
       'meals', 'pain', 'psych1', 'psych4',
       'psych5', 'psych6', 'reflex', 'urine',
       'disability', 'dnr', 'primary']
    

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

    for weight in weightings:
        for key in weightings[weight]:
            df.loc[df[weight] == key, weight] = weightings[weight][key]
    


    # # col_to_kee
    df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    # drop NaN
    # df.dropna(inplace=True)
    print(len(df), df.keys())

    return df
    
def split_feature_label(df: pd.DataFrame):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4, random_state=42)

    # tensor error
    X_train = np.asarray(X_train).astype(np.float64)
    X_val = np.asarray(X_val).astype(np.float64)
    X_test = np.asarray(X_test).astype(np.float64)
    y_train = np.asarray(y_train).astype(np.float64)
    y_val = np.asarray(y_val).astype(np.float64)
    y_test = np.asarray(y_test).astype(np.float64)
    print(X_train.shape)


    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.0075)),      # Another hidden layer with 64 neurons and ReLU activation
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=12, batch_size=48, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.keras')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    # import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    # plt.show()

if __name__ == "__main__":
    # data_path = './THT_OUT.csv'
    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    train_model(X, y)
    