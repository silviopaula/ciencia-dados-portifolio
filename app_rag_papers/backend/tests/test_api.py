"""Suíte de testes automatizados (unittest + TestClient) da API PaperRAG Studio."""

from __future__ import annotations

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, engine


class TestPaperRAGAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_01_health_and_demo_autoload(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertGreaterEqual(data["documents_loaded"], 2)
        self.assertGreater(data["chunks_indexed"], 100)

    def test_02_library_and_chunk_stats(self) -> None:
        resp = self.client.get("/api/library")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["documents"]), 2)
        stats = data["stats"]
        self.assertEqual(stats["total_pages"], 85)
        self.assertGreater(stats["vocab_size"], 1000)
        self.assertGreater(stats["lsa_dimensions"], 10)
        self.assertGreater(len(stats["histogram"]), 2)
        self.assertGreaterEqual(len(data["suggested_questions"]), 4)

    def test_03_chunks_pagination_and_search(self) -> None:
        resp = self.client.get("/api/chunks", params={"search": "parallel trends", "limit": 10})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["total"], 10)
        self.assertLessEqual(len(data["items"]), 10)
        self.assertTrue(any(item["overlap_prefix_chars"] > 0 for item in data["items"]))

    def test_04_bilingual_hybrid_query_continuous_treatment(self) -> None:
        resp = self.client.post(
            "/api/query",
            json={
                "question": "Qual a diferença entre Tendências Paralelas padrão e Tendências Paralelas Fortes no tratamento contínuo?",
                "top_k": 5,
                "alpha": 0.55,
                "use_mmr": True,
                "provider": "extractive",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["provider_used"], "extractive")
        self.assertEqual(len(data["citations"]), 5)
        self.assertFalse(data["diagnostics"]["out_of_scope_warning"])
        self.assertGreaterEqual(data["diagnostics"]["grounding_score"], 70.0)
        self.assertIn("strong", data["diagnostics"]["expanded_terms"])
        self.assertIn("parallel", data["diagnostics"]["expanded_terms"])
        # O chunk #1 deve vir do artigo de Continuous Treatment
        self.assertIn("Continuous Treatment", data["citations"][0]["document_title"])
        self.assertIn("[1]", data["answer"])
        self.assertGreater(data["prompt_inspection"]["estimated_tokens"], 200)

    def test_05_bilingual_query_multiple_time_periods(self) -> None:
        resp = self.client.post(
            "/api/query",
            json={
                "question": "Como é definido o group-time average treatment effect ATT(g,t) em múltiplos períodos?",
                "top_k": 5,
                "alpha": 0.5,
                "use_mmr": True,
                "provider": "extractive",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["diagnostics"]["out_of_scope_warning"])
        self.assertTrue(
            any("Multiple Time Periods" in c["document_title"] for c in data["citations"][:2])
        )

    def test_06_out_of_scope_anti_hallucination_guardrail(self) -> None:
        resp = self.client.post(
            "/api/query",
            json={
                "question": "Qual foi a receita líquida da Petrobras no terceiro trimestre de 2025 e a política de dividendos?",
                "top_k": 5,
                "alpha": 0.55,
                "use_mmr": True,
                "provider": "extractive",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["diagnostics"]["out_of_scope_warning"])
        self.assertEqual(data["diagnostics"]["confidence_level"], "baixa")
        self.assertIn("Anti-Alucinação", data["answer"])

    def test_07_rebuild_chunk_config_and_restore(self) -> None:
        r1 = self.client.post("/api/index/rebuild", json={"chunk_size": 700, "chunk_overlap": 120})
        self.assertEqual(r1.status_code, 200)
        chunks_700 = r1.json()["stats"]["total_chunks"]

        r2 = self.client.post("/api/index/rebuild", json={"chunk_size": 1000, "chunk_overlap": 180})
        self.assertEqual(r2.status_code, 200)
        chunks_1000 = r2.json()["stats"]["total_chunks"]
        self.assertGreater(chunks_700, chunks_1000)

    def test_08_providers_endpoint(self) -> None:
        resp = self.client.get("/api/providers")
        self.assertEqual(resp.status_code, 200)
        ids = {p["id"] for p in resp.json()["providers"]}
        self.assertEqual(ids, {"extractive", "ollama", "openai", "gemini"})

    def test_09_upload_and_delete_roundtrip(self) -> None:
        demo_pdf = next(Path(engine.demo_papers_dir).glob("*.pdf"))
        pdf_bytes = demo_pdf.read_bytes()
        upload_resp = self.client.post(
            "/api/documents/upload",
            files={"files": ("copia_teste_upload.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(upload_resp.status_code, 200)
        docs = upload_resp.json()["documents"]
        uploaded = [d for d in docs if d["filename"] == "copia_teste_upload.pdf"]
        self.assertEqual(len(uploaded), 1)

        del_resp = self.client.delete(f"/api/documents/{uploaded[0]['doc_id']}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(len(del_resp.json()["documents"]), 2)


if __name__ == "__main__":
    unittest.main()
