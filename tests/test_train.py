
def test_train_main_runs_fast(monkeypatch, tmp_path, df_minimo):
    import src.train as tr

    monkeypatch.setattr(tr, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(tr, "MODEL_PATH", tmp_path / "pipeline.joblib")
    monkeypatch.setattr(tr, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(tr, "TEST_PATH", tmp_path / "test.csv")

    df = df_minimo.copy()
    if tr.TARGET_COL not in df.columns:
        df[tr.TARGET_COL] = 0

    # dataset pequeno
    monkeypatch.setattr(tr, "DF_2024", df_minimo.copy())

    tr.main()

    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "test.csv").exists()
