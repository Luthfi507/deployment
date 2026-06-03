import mlflow

def test_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test_experiment")
    
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.85)
        mlflow.log_artifact("testing.py")

if __name__ == "__main__":
    test_mlflow()