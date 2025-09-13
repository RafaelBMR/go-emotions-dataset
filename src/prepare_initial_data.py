from collections import Counter
import os
import json
import argparse

import pandas as pd
import numpy as np


def clean_dataframe_labels(df, label2id):
    """
        Creates a new dataframe copying the samples, but removing the "neutral" label
    """
    new_df = []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        if label2id['neutral'] in row_dict['labels']:
            row_dict['labels'].remove(label2id['neutral'])
        new_df.append(row_dict)
    return pd.DataFrame(new_df)


def define_sample_importance(dataset_size, sample_labels, id2label, classes_count):
    """
        The importance is inversely proportional to the frequency of the least frequent label of the sample.
    """
    least_frequent_class = None
    least_frequent_occurrences = dataset_size+1
    for label_id in sample_labels:
        label = id2label[label_id]
        if classes_count[label] < least_frequent_occurrences:
            least_frequent_class = label
            least_frequent_occurrences = classes_count[label]
    return dataset_size/least_frequent_occurrences


def process_data(args):
    # Loading original data
    with open(os.path.join(args.input_data_path, "id2label.json")) as f:
        id2label = json.loads(f.read())
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    df_train = pd.read_json(os.path.join(args.input_data_path, "train.json"))
    df_validation = pd.read_json(os.path.join(args.input_data_path, "validation.json"))
    df_test = pd.read_json(os.path.join(args.input_data_path, "test.json"))

    # Counting occurrences of each class in training data
    classes_count = Counter()
    for i, row in df_train.iterrows():
        for label_id in row['labels']:
            classes_count[id2label[label_id]] += 1

    # Determining the importance of each sample.
    # Samples that contain a low frequency label are more important than samples that contain
    # only high frequency labels
    samples_importances = []
    for i, row in df_train.iterrows():
        sample_importance = define_sample_importance(len(df_train), row['labels'], id2label, classes_count)
        samples_importances.append(sample_importance)

    # Sorting samples indices by importance and splitting training data in two.
    # Half for training the initial classifier, and half for later usage
    
    # Setting seed to ensure reproducibility
    np.random.seed(42)
    # Converting samples importances to a probability distribution, so we can use
    # Numpy's random.choice function.
    samples_importances = np.array(samples_importances) / np.sum(samples_importances)
    sorted_indices = np.random.choice(a=range(len(df_train)), 
                                      size=len(df_train),
                                      replace=False,
                                      p=samples_importances)

    initial_model_indices = sorted_indices[:len(df_train)//2]
    later_usage_indices = sorted_indices[len(df_train)//2:]

    df_train_initial_model = df_train.iloc[initial_model_indices]
    df_train_later_usage = df_train.iloc[later_usage_indices]

    # Preparing output data
    # Removing "Neutral" from the dataset annotations and label mapping
    df_train_initial_model = clean_dataframe_labels(df_train_initial_model, label2id)
    df_train_later_usage = clean_dataframe_labels(df_train_later_usage, label2id)
    df_validation = clean_dataframe_labels(df_validation, label2id)
    df_test = clean_dataframe_labels(df_test, label2id)
    # Removing neutral class from label mapping
    id2label.pop(27)

    # Saving output data
    os.makedirs(args.output_data_path, exist_ok=True)

    df_train_initial_model.to_json(os.path.join(args.output_data_path, "train_initial_model.json"), orient='records')
    df_train_later_usage.to_json(os.path.join(args.output_data_path, "train_later_usage.json"), orient='records')

    df_validation.to_json(os.path.join(args.output_data_path, "validation.json"), orient='records')
    df_test.to_json(os.path.join(args.output_data_path, "test.json"), orient='records')

    with open(os.path.join(args.output_data_path, "id2label.json"), mode='w') as f:
        f.write(json.dumps(id2label))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_path")
    parser.add_argument("--output_data_path")

    args = parser.parse_args()

    process_data(args)
    