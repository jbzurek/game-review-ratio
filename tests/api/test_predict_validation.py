def test_predict_validation_error_422(api_client):
    c, _, _ = api_client

    r = c.post("/predict", json={"required_age": "oops"})
    assert r.status_code == 422
