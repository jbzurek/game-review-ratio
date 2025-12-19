from fastapi.testclient import TestClient
from src.api.main import app, database
client = TestClient(app)

# testuje walidacje Pydantic
def test_validation_error():
    r = client.post("/predict", json={"feature1": 1.23, "feature2": 4.56})
    assert r.status_code == 422

# testuje poprawność zapisu rekordu do bazy danych
async def test_predict_returns_200_and_saves_record_in_db(db_session):

    query = """SELECT IF EXIST COUNT(*) predictions"""

    count_before = await database.execute(query=query)

    r = client.post("/predict", json={"feature1": "test", "feature2": 4.56})

    assert r.status_code == 200
    assert await database.execute(query=query) == count_before + 1

#health test
def test_health_check():
    r = client.get("/health")
    assert r == {"status": "ok"}