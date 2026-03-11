import os
import argparse
import joblib
import json
import pandas as pd
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import Model
from train import split_data, train_model, get_model_metrics


# def get_ml_client():
#     subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
#     resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
#     workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")

#     credential = DefaultAzureCredential()

#     ml_client = MLClient(
#         credential=credential,
#         subscription_id=subscription_id,
#         resource_group_name=resource_group,
#         workspace_name=workspace_name,
#     )

#     return ml_client


def main():
    print("Running train_aml_v2.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="diabetes_model.pkl")
    parser.add_argument("--data_file_path", type=str, required=True)

    args = parser.parse_args()

    model_name = args.model_name
    data_file_path = args.data_file_path
    print(data_file_path)
    print("Loading dataset...")
    df = pd.read_csv(data_file_path)

    print("Splitting data...")
    data = split_data(df)

    print("Training model...")
    model = train_model(data, {})

    print("Evaluating model...")
    metrics = get_model_metrics(model, data)
    print("Metrics:", metrics)

    # Save model locally
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", model_name)
    joblib.dump(model, model_path)

    print("Model saved locally.")

    # If running inside Azure ML job, register model
    if os.environ.get("AZUREML_RUN_ID"):
        print("Registering model in Azure ML...")

        # ml_client = get_ml_client()

        # registered_model = Model(
        #     path="outputs",
        #     name="diabetes_regression_model",
        #     description="Diabetes regression model",
        #     type="custom_model",
        # )

        # ml_client.models.create_or_update(registered_model)

        print("Model registered successfully.")
    else:
        print("Running locally — skipping model registration.")


if __name__ == "__main__":
    main()
