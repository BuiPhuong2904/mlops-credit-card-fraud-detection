import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    os.environ["AZ_SUBSCRIPTION_ID"],
    "rg-fraud-mlops",
    "mlw-fraud-detection",
)

cluster = AmlCompute(
    name="cpu-cluster",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=1,
    idle_time_before_scale_down=120,
)
ml_client.compute.begin_create_or_update(cluster).result()
print("Cluster ready:", cluster.name)