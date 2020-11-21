import os
from src.mbs.utils.log_helper import get_logger

from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment

from src.mbs.utils.aml_interface import AMLInterface
from src.mbs.utils.constants import *

logger = get_logger("create_aml_environment")


def get_dist_dir():
    __here__ = os.path.dirname(__file__)
    dist_dir = os.path.join(__here__, '../../', 'dist')
    return dist_dir


def retrieve_whl_filepath():
    dist_dir = get_dist_dir()
    if not os.path.isdir(dist_dir):
        raise FileNotFoundError("Couldn't find dist directory")
    dist_files = os.listdir(dist_dir)
    whl_file = [
        f for f in dist_files
        if f.startswith('mbs')
           and f.endswith('whl')
    ]
    if not whl_file:
        raise FileNotFoundError("Couldn't find wheel file")
    return os.path.join(dist_dir, whl_file[0])


def create_aml_environment(aml_interface):
    aml_env = Environment(name=AML_ENVIRONMENT_NAME)
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package("numpy==1.18.2")
    conda_dep.add_pip_package("pandas==1.0.3")
    conda_dep.add_pip_package("scikit-learn==0.22.2.post1")
    conda_dep.add_pip_package("joblib==0.14.1")
    conda_dep.add_pip_package("azure-storage-blob==12.3.0")

    aml_env.environment_variables[AZURE_STORAGE_ACCOUNT_NAME] = os.getenv(AZURE_STORAGE_ACCOUNT_NAME)
    aml_env.environment_variables[AZURE_STORAGE_ACCOUNT_KEY] = os.getenv(AZURE_STORAGE_ACCOUNT_KEY)
    aml_env.environment_variables[MODEL_NAME_VARIABLE] = MODEL_NAME

    logger.info(f"set environment variables on compute environment: {aml_env.environment_variables}")

    whl_filepath = retrieve_whl_filepath()
    whl_url = Environment.add_private_pip_wheel(
        workspace=aml_interface.workspace,
        file_path=whl_filepath,
        exist_ok=True
    )
    conda_dep.add_pip_package(whl_url)
    aml_env.python.conda_dependencies = conda_dep
    aml_env.docker.enabled = True
    return aml_env


def main():
    # Retrieve vars from env
    workspace_name = os.environ['AML_WORKSPACE_NAME']
    resource_group = os.environ['RESOURCE_GROUP']
    subscription_id = os.environ['SUBSCRIPTION_ID']

    storage_account_name = os.environ[AZURE_STORAGE_ACCOUNT_NAME]
    storage_account_key = os.environ[AZURE_STORAGE_ACCOUNT_KEY]

    spn_credentials = {
        'tenant_id': os.environ['TENANT_ID'],
        'service_principal_id': os.environ['SPN_ID'],
        'service_principal_password': os.environ['SPN_PASSWORD'],
    }

    aml_interface = AMLInterface(
        spn_credentials, subscription_id, workspace_name, resource_group
    )

    aml_env = create_aml_environment(aml_interface)
    logger.info(f"Succesfully created Azure ML Environment {aml_env.name}. Version: {aml_env.version}")

    aml_interface.register_aml_environment(aml_env)
    logger.info(f"Succesfully registered Azure ML Environment {aml_env.name}. Version: {aml_env.version}")

    aml_interface.register_datastore(TRAINING_DATASTORE, TRAINING_CONTAINER, storage_account_name, storage_account_key)
    logger.info(f"Succesfully registered Dataset Store: {TRAINING_DATASTORE} "
        f"from {TRAINING_CONTAINER}@{storage_account_name}.blob.core.windows.net")





if __name__ == '__main__':
    main()
