import torch
import mlflow
from ruamel.yaml import YAML

with open("params.yaml") as f:
    yaml = YAML(typ='safe')
    params = yaml.load(f)

BEST_MODEL_RUN_ID = params["best_model"]["run_id"]

TRACKING_SERVER_HOST = 'localhost'
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

model = mlflow.pytorch.load_model(f"runs:/{BEST_MODEL_RUN_ID}/model")

torch.save(model.state_dict(), "model.pth")
