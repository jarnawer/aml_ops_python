import os
from io import StringIO
from src.mbs.utils.constants import *
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
import pandas as pd


def __get_blob_storage_client() -> BlobServiceClient:
    account_key = os.environ.get(AZURE_STORAGE_ACCOUNT_KEY)
    account_name = os.environ.get(AZURE_STORAGE_ACCOUNT_NAME)
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
    return blob_service_client


def get_data():
    blob_storage_client = __get_blob_storage_client()
    blob_client = blob_storage_client.get_blob_client("raw", "iris/year=2020/month=10/day=05/iris.csv")
    content = blob_client.download_blob().readall().decode()
    dataframe = pd.read_csv(StringIO(content))
    x = dataframe.values[:, 0:4]
    y = dataframe.values[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    get_data()
