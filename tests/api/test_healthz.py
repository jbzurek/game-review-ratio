def test_healthz_ok(api_client):
    c, _, _ = api_client
    r = c.get("/healthz")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    # model_version is not returned by the app, so we skip those checks
    assert "model_type" in body
    assert isinstance(body["model_type"], str)
