import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''

    ## Create a group by using the first part of the 'Non-proprietary Name' field
    ndc_code_df = ndc_df.copy()
    ndc_code_df['GroupProp'] = ndc_code_df['Non-proprietary Name'].str.split(' ', expand = True).iloc[:, 0]
    
    # Clean up the group names
    ndc_code_df['GroupProp'] = ndc_code_df['GroupProp'] \
        .replace({'Glyburide-metformin': 'Glyburide', 'Human': 'Insulin', 'Pioglitazone': 'Pioglitazole'})

    # Create a dictionary to map a NDC code with a newly created group
    ndc_map = dict()
    for index, row in ndc_code_df.iterrows():
        ndc_map[row['NDC_Code']] = row['GroupProp']
        
    # Create a new column 'generic_drug_name' and return the updated dataframe
    df = df.copy()
    df['generic_drug_name'] = df['ndc_code'].apply(lambda x: ndc_map[x] if(pd.notnull(x)) else x)
    
    # Drop the 'ndc_code' column since we already have the 'generic_drug_name' column
    df = df.drop('ndc_code', axis='columns')
    
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    # Define a list of interested columns
    col_grps = ['encounter_id', 'patient_nbr']
    col_others = ['generic_drug_name']
    
    # Then, create an encounter level dataframe and sorted ascendingly by the 'encounter_id' column
    encoder_df = df[col_grps + col_others].groupby(col_grps)[col_others].count().reset_index()
    encoder_df = encoder_df.sort_values('encounter_id')
    
    # Get the first encounter id of each patient
    first_encounter_values = encoder_df.groupby(['patient_nbr'])['encounter_id'].head(1).values
    
    # Create a new data frame with only the first encounter for a given patient
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_values)]
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    key = 'patient_nbr'
    df = df.iloc[np.random.permutation(len(df))]
    
    # Approximately 60%/20%/20% train/validation/test split

    val_percentage = 0.2
    test_percentage = 0.2

    unique_values = df[key].unique()
    total_values = len(unique_values)
    sample_size_train = round(total_values * (1 - (val_percentage + test_percentage) ))
    sample_size_val = round(total_values * val_percentage)

    train = df[df[key].isin(unique_values[:sample_size_train])].reset_index(drop=True)
    validation = df[df[key].isin(unique_values[sample_size_train:sample_size_train+sample_size_val])].reset_index(drop=True)
    test = df[df[key].isin(unique_values[sample_size_train+sample_size_val:])].reset_index(drop=True)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        one_hot_principal_diagnosis_feature = tf.feature_column.indicator_column(tf_categorical_feature_column)
        
        output_tf_list.append(one_hot_principal_diagnosis_feature)

    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)

    tf_numeric_feature = tf.feature_column.numeric_column(
            key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    # Since our clinical trical requires administering the drug over at least 5-7 days of time in the hospital,
    # We will output a value 1 if the mean predication value is greater than 5
    student_binary_prediction = np.array((df[col] >= 5).astype(int))
    
    return student_binary_prediction
