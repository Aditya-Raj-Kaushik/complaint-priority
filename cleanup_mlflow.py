import mlflow
import dagshub

dagshub.init(
    repo_owner="Aditya-Raj-Kaushik",
    repo_name="complaint-priority",
    mlflow=True
)

client = mlflow.tracking.MlflowClient()

EXP_NAME = "complaint_priority_training"

exp = client.get_experiment_by_name(EXP_NAME)
if exp:
    client.delete_experiment(exp.experiment_id)
    print(f"Deleted experiment: {EXP_NAME}")
else:
    print(f"Experiment not found: {EXP_NAME}")
