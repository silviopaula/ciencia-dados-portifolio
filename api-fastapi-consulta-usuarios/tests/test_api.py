"""Suíte de testes automatizados da API de Análise de Usuários.

Compatível tanto com `pytest` quanto com `python -m unittest discover -s tests`.
Valida regras de negócio em um payload controlado e também testa o carregamento
do dataset real (data/usuarios.json).
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.services import DEFAULT_DATA_PATH, repository


SAMPLE_USERS = [
    {
        "id": "u-1",
        "name": "Alice Silva",
        "age": 30,
        "score": 950,
        "active": True,
        "country": "Brasil",
        "team": {
            "name": "Data Core",
            "leader": True,
            "projects": [
                {"name": "Lakehouse", "completed": True},
                {"name": "Feature Store", "completed": False},
            ],
        },
        "logs": [
            {"date": "2025-03-25", "action": "login"},
            {"date": "2025-03-25", "action": "logout"},
            {"date": "2025-03-26", "action": "login"},
        ],
    },
    {
        "id": "u-2",
        "name": "Bruno Costa",
        "age": 28,
        "score": 910,
        "active": False,  # Inativo -> NÃO entra em /superusers
        "country": "Brasil",
        "team": {
            "name": "Data Core",
            "leader": False,
            "projects": [{"name": "ETL", "completed": True}],
        },
        "logs": [{"date": "2025-03-25", "action": "login"}],
    },
    {
        "id": "u-3",
        "name": "Carla Mendes",
        "age": 35,
        "score": 920,
        "active": True,
        "country": "Portugal",
        "team": {
            "name": "AI Labs",
            "leader": True,
            "projects": [{"name": "RAG Assistant", "completed": True}],
        },
        "logs": [
            {"date": "2025-03-25", "action": "login"},
            {"date": "2025-03-27", "action": "login"},
        ],
    },
    {
        "id": "u-4",
        "name": "Daniel Rocha",
        "age": 24,
        "score": 700,  # Score < 900 -> NÃO entra em /superusers
        "active": True,
        "country": "Brasil",
        "team": {
            "name": "AI Labs",
            "leader": False,
            "projects": [],
        },
        "logs": [],
    },
]


class TestUserAnalyticsAPI(unittest.TestCase):
    """Testes funcionais dos endpoints usando um conjunto controlado de usuários."""

    def setUp(self) -> None:
        repository.set_users(SAMPLE_USERS)
        self.client = TestClient(app)

    def test_health_check(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "online")
        self.assertEqual(body["users_loaded"], 4)

    def test_post_users_ingestion(self) -> None:
        resp = self.client.post("/users", json=SAMPLE_USERS[:2])
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["user_count"], 2)
        self.assertGreaterEqual(body["processing_time_ms"], 0.0)

    def test_superusers_filter_and_pagination(self) -> None:
        resp = self.client.get("/superusers?limit=1&offset=0")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # Apenas Alice (950, ativo) e Carla (920, ativo) qualificam
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["returned"], 1)
        self.assertEqual(body["data"][0]["id"], "u-1")

    def test_top_countries(self) -> None:
        resp = self.client.get("/top-countries?top_n=5")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        counts = {item["country"]: item["total"] for item in body["data"]}
        self.assertEqual(counts, {"Brasil": 1, "Portugal": 1})

    def test_team_insights(self) -> None:
        resp = self.client.get("/team-insights")
        self.assertEqual(resp.status_code, 200)
        teams = {item["team"]: item for item in resp.json()["data"]}

        self.assertEqual(teams["Data Core"]["total_members"], 2)
        self.assertEqual(teams["Data Core"]["leaders"], 1)
        self.assertEqual(teams["Data Core"]["completed_projects"], 2)
        self.assertEqual(teams["Data Core"]["active_percentage"], 50.0)

        self.assertEqual(teams["AI Labs"]["total_members"], 2)
        self.assertEqual(teams["AI Labs"]["leaders"], 1)
        self.assertEqual(teams["AI Labs"]["completed_projects"], 1)
        self.assertEqual(teams["AI Labs"]["active_percentage"], 100.0)

    def test_active_users_per_day_and_min_filter(self) -> None:
        resp_all = self.client.get("/active-users-per-day")
        self.assertEqual(resp_all.status_code, 200)
        by_day = {item["date"]: item["total"] for item in resp_all.json()["data"]}
        self.assertEqual(by_day, {"2025-03-25": 3, "2025-03-26": 1, "2025-03-27": 1})

        resp_min = self.client.get("/active-users-per-day?min=2")
        self.assertEqual(resp_min.status_code, 200)
        self.assertEqual(resp_min.json()["data"], [{"date": "2025-03-25", "total": 3}])

    def test_evaluation_endpoint(self) -> None:
        resp = self.client.get("/evaluation")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["score_percentage"], 100.0)
        self.assertEqual(body["passed"], body["total"])

    def test_real_dataset_integration(self) -> None:
        """Valida o carregamento e consulta sobre o dataset real (data/usuarios.json)."""
        if not DEFAULT_DATA_PATH.exists():
            self.skipTest("Dataset real data/usuarios.json não disponível.")

        count = repository.load_from_file(DEFAULT_DATA_PATH)
        self.assertGreater(count, 0)

        eval_resp = self.client.get("/evaluation")
        self.assertEqual(eval_resp.status_code, 200)
        self.assertEqual(eval_resp.json()["score_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
