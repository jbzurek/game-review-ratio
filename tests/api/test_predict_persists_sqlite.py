def test_predict_returns_200_and_persists_to_db(api_client, count_predictions):
    c, db_path, _ = api_client

    payload = {
        "required_age": 0,
        "price": 19.99,
        "dlc_count": 0,
        "windows": True,
        "mac": False,
        "linux": False,
        "metacritic_score": 0,
        "achievements": 0,
        "discount": 0,
        "release_year": 2024,
        "release_month": 6,
        "genres": ["Action"],
        "categories": ["Single-player"],
        "tags": ["Indie"],
        "developers": [],
        "publishers": [],
        "supported_languages": ["English"],
        "full_audio_languages": ["English"],
    }

    before = count_predictions(db_path)

    r = c.post("/predict", json=payload)
    assert r.status_code == 200

    body = r.json()
    assert "prediction" in body
    assert "model_version" in body

    after = count_predictions(db_path)
    assert after == before + 1
