import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = os.environ["AZ_SUBSCRIPTION_ID"]
RESOURCE_GROUP = "rg-fraud-mlops"
LOCATION = "eastasia"
WORKSPACE_NAME = "mlw-fraud-detection"

ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP)

ws = Workspace(
    name=WORKSPACE_NAME,
    location=LOCATION,
    display_name="Fraud Detection MLOps",
    description="Workspace cho do an MLOps phat hien gian lan",
    tags={"project": "fraud-mlops"},
)
ws = ml_client.workspaces.begin_create(ws).result()
print("Workspace created:", ws.name)