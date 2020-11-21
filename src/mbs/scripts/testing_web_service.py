import json
import os
import csv
import numpy as np
import pandas as pd
import requests
from azureml.core.webservice import Webservice
from sklearn.metrics import f1_score, accuracy_score
import sys

sys.path.append("../utils/")
from mbs.utils.constants import AML_ENVIRONMENT_NAME, DEPLOYMENT_SERVICE_NAME, MODEL_NAME

URI = "http://f57ca1a4-f8e0-4a77-b955-a2d1e1f5e6cd.northeurope.azurecontainer.io/score"


def get_validation_data():
    df = pd.read_csv("../../../data/iris_test.csv", )
    df = df.sample(frac=0.1, replace=False, random_state=1)

    x_valid = df.drop(["Species"], axis=1)
    y_valid = df["Species"]

    return x_valid, y_valid


def get_web_service_uri(aml_interface):
    service = Webservice(
        name=DEPLOYMENT_SERVICE_NAME,
        workspace=aml_interface.workspace
    )
    return service.scoring_uri


def make_predictions(x_df, scoring_uri):
    data = json.dumps({'data': x_df.values.tolist()})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(scoring_uri, data=data, headers=headers)
    return np.array(response.json())


def score_predictions(y_valid, y_pred):
    model_validation_f1_score = round(f1_score(y_valid, y_pred), 3)
    print(f"F1 Score on Validation Data Set: {model_validation_f1_score}")


def main():
    x_valid, y_valid = get_validation_data()
    # x_valid = remove_collinear_cols(x_valid)
    # y_valid = y_valid['Target']
    # aml_interface = AMLInterface(
    #    spn_credentials, subscription_id, workspace_name, resource_group
    # )
    # scoring_uri = get_web_service_uri(aml_interface)
    y_pred = make_predictions(x_valid, URI)
    # score_predictions(y_valid, y_pred)
    accuracy_score(y_valid, y_pred)


if __name__ == '__main__':
    main()
