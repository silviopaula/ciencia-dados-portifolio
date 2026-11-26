"""Suíte de testes automatizados (unittest + TestClient) da API MacroRAG Tutor Studio."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestMacroRAGAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_01_health_and_heijdra_autoload(self) -> None:
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        hd = h.json()
        self.assertEqual(hd["app_name"], "MacroRAG Tutor Studio API")
        self.assertGreaterEqual(hd["documents_loaded"], 1)
        self.assertGreaterEqual(hd["chapters_indexed"], 15)
        self.assertGreater(hd["chunks_indexed"], 1000)

    def test_02_library_chapters_and_keywords(self) -> None:
        lib = self.client.get("/api/library")
        self.assertEqual(lib.status_code, 200)
        data = lib.json()
        self.assertGreaterEqual(len(data["chapters"]), 15)
        self.assertTrue(any("New Keynesian" in ch["title"] for ch in data["chapters"]))
        self.assertTrue(any("Endogenous economic growth" in ch["title"] for ch in data["chapters"]))

    def test_03_bilingual_macro_tutor_ricardian_equivalence(self) -> None:
        res = self.client.post(
            "/api/query",
            json={
                "question": "O que é a Equivalência Ricardiana e como o déficit público afeta a demanda agregada?",
                "study_mode": "tutor",
                "top_k": 5,
                "alpha": 0.5,
                "provider": "extractive",
            },
        )
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertFalse(d["diagnostics"]["out_of_scope_warning"])
        self.assertGreaterEqual(d["diagnostics"]["grounding_score"], 75.0)
        self.assertTrue(d["privacy"]["is_fully_local"])
        self.assertEqual(d["privacy"]["external_sent_chars"], 0)
        # Deve citar Capítulo 5 (The government budget deficit)
        self.assertTrue(any("budget deficit" in c["chapter_title"].lower() for c in d["citations"][:3]))

    def test_04_compare_mode_new_keynesian_vs_rbc(self) -> None:
        res = self.client.post(
            "/api/query",
            json={
                "question": "Compare o modelo Novo Keynesiano com rigidez de preços versus Ciclos Reais de Negócios (RBC).",
                "study_mode": "compare",
                "top_k": 6,
                "alpha": 0.5,
                "provider": "extractive",
            },
        )
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertFalse(d["diagnostics"]["out_of_scope_warning"])
        self.assertGreaterEqual(d["diagnostics"]["unique_chapters_cited"], 2)

    def test_05_out_of_scope_guardrail(self) -> None:
        res = self.client.post(
            "/api/query",
            json={
                "question": "Qual foi a escalação titular da Seleção Brasileira na final da Copa do Mundo de 2026?",
                "study_mode": "verify",
                "top_k": 5,
                "provider": "extractive",
            },
        )
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertTrue(d["diagnostics"]["out_of_scope_warning"])
        self.assertEqual(d["diagnostics"]["confidence_level"], "baixa")

    def test_06_chapter_flashcards_generation(self) -> None:
        lib = self.client.get("/api/library").json()
        nk_ch = next(ch for ch in lib["chapters"] if "New Keynesian" in ch["title"])
        fc = self.client.post(
            "/api/flashcards",
            json={"chapter_id": nk_ch["chapter_id"], "count": 4, "provider": "extractive"},
        )
        self.assertEqual(fc.status_code, 200)
        data = fc.json()
        self.assertEqual(len(data["cards"]), 4)
        self.assertTrue(all("New Keynesian" in c["chapter_title"] for c in data["cards"]))

    def test_07_upload_markdown_lecture_notes_and_delete(self) -> None:
        md_content = (
            "# Politica Monetaria no Brasil\n\n"
            "O Comite de Politica Monetaria (COPOM) define a meta da taxa Selic utilizando o regime de metas para a inflacao "
            "baseado no IPCA, ancorando expectativas inflacionarias e controlando o hiato do produto.\n\n"
            "## Regra de Taylor\n\n"
            "A Regra de Taylor relaciona a taxa basica de juros nominal a taxa neutra real, aos desvios da inflacao esperada "
            "em relacao a meta e ao hiato do produto da economia brasileira."
        )
        up = self.client.post(
            "/api/documents/upload",
            files={"files": ("notas_copom_brasil.md", md_content.encode("utf-8"), "text/markdown")},
        )
        self.assertEqual(up.status_code, 200)
        docs = up.json()["documents"]
        added = [d for d in docs if d["filename"] == "notas_copom_brasil.md"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["chapters_count"], 2)

        # Consulta direcionada às notas enviadas
        q = self.client.post(
            "/api/query",
            json={
                "question": "Como a Regra de Taylor funciona no COPOM com a taxa Selic e o IPCA?",
                "doc_ids": [added[0]["doc_id"]],
                "top_k": 2,
                "provider": "extractive",
            },
        )
        self.assertEqual(q.status_code, 200)
        self.assertFalse(q.json()["diagnostics"]["out_of_scope_warning"])

        # Remove documento de teste
        dl = self.client.delete(f"/api/documents/{added[0]['doc_id']}")
        self.assertEqual(dl.status_code, 200)
        self.assertFalse(any(d["doc_id"] == added[0]["doc_id"] for d in dl.json()["documents"]))


if __name__ == "__main__":
    unittest.main()
