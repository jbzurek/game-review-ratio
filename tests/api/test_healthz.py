def test_healthz_ok(api_client):
    c, _, _ = api_client
    r = c.get("/healthz")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert isinstance(body["model_version"], str)
    assert len(body["model_version"]) > 0
