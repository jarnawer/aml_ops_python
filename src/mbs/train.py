import os
import joblib
from azureml.core import Datastore, Dataset, Run
from azureml.data.abstract_dataset import AbstractDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from io import StringIO
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
import pandas as pd

__here__ = os.path.dirname(__file__)


def __get_blob_storage_client() -> BlobServiceClient:
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_url = f"https://{account_name}.blob.core.windows.net"
    print(f"get data from {account_url}")
    blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
    return blob_service_client


def get_data(workspace):
    # blob_storage_client = __get_blob_storage_client()
    # blob_client = blob_storage_client.get_blob_client("raw", "iris/year=2020/month=10/day=05/iris.csv")
    datastore = Datastore.get(workspace, "train")
    datastore_path = [(datastore, "iris/year=2020/month=10/day=05/iris.csv")]
    dataset = Dataset.Tabular.from_delimited_files(path=datastore_path)
    # content = blob_client.download_blob().readall().decode()
    # dataframe = pd.read_csv(StringIO(content))
    dataframe = dataset.to_pandas_dataframe()
    x = dataframe.values[:, 0:4]
    y = dataframe.values[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, y_train, x_test, y_test, dataset


def train_model(x_train, y_train):
    classifier = LogisticRegression(multi_class="ovr")
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(classifier, x_test, y_test, run):
    y_pred = classifier.predict(x_test)
    model_f1_score = f1_score(y_test, y_pred, average="weighted")
    model_accuracy_score = accuracy_score(y_test, y_pred)
    run.log('F1_Score', model_f1_score)
    run.log('Accuracy', model_accuracy_score)


def save_model(classifer):
    output_dir = os.path.join(__here__, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(classifer, model_path)
    return model_path


def register_model(run, model_path, dataset: AbstractDataset):
    run.upload_file(model_path, "outputs/model.pkl")
    model = run.register_model(
        model_name="iris_model",
        model_path="outputs/model.pkl",
        datasets=[(dataset.name, dataset)]
    )
    run.log('Model_ID', model.id)


def main():
    run = Run.get_context()
    workspace = run.experiment.workspace
    x_train, y_train, x_test, y_test, dataset = get_data(workspace)
    classifier = train_model(x_train, y_train)
    evaluate_model(classifier, x_test, y_test, run)
    model_path = save_model(classifier)
    register_model(run, model_path, dataset)


if __name__ == '__main__':
    main()
