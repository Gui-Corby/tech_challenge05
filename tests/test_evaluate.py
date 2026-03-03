import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier

def test_evaluate_main_writes_metrics(tmp_path, monkeypatch):
    import src.evaluate as ev

    # 1) paths temporários
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "pipeline.joblib"
    test_path = artifacts_dir / "test.csv"
    eval_metrics_path = artifacts_dir / "metrics_eval.json"

    # 2) cria um modelo simples (picklable) e salva
    X_train = pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [1, 1, 0, 0]})
    y_train = pd.Series([0, 1, 0, 1], name="Defasagem")

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    # 3) cria test.csv com a coluna TARGET_COL
    test_df = X_train.copy()
    test_df[ev.TARGET_COL] = y_train
    test_df.to_csv(test_path, index=False)

    # 4) monkeypatch no módulo evaluate
    monkeypatch.setattr(ev, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ev, "MODEL_PATH", model_path)
    monkeypatch.setattr(ev, "TEST_PATH", test_path)
    monkeypatch.setattr(ev, "EVAL_METRICS_PATH", eval_metrics_path)

    # 5) roda
    ev.main()

    # 6) valida que salvou
    assert eval_metrics_path.exists()
