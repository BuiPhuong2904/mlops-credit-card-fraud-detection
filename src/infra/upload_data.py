import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    os.environ["AZ_SUBSCRIPTION_ID"],
    "rg-fraud-mlops",
    "mlw-fraud-detection",
)

data = Data(
    name="creditcard-raw",
    version="1",
    type=AssetTypes.URI_FILE,
    path="data/creditcard.csv",
    description="Credit Card Fraud Detection dataset (Kaggle mlg-ulb)",
)
ml_client.data.create_or_update(data)
print("Data uploaded")