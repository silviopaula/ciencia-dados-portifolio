"""Testes unitários do pipeline NLP & RAG do projeto avaliacoes-ecommerce-olist-NLP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from nlp_pipeline import (
    CustomerSupportRAG,
    build_text_vectorizer,
    chronological_split,
    clean_pt_text,
    compute_lexicon_score,
    detect_aspects,
    enrich_text_signals,
    evaluate_multiclass,
)


class TestNLPEcommercePipeline(unittest.TestCase):
    def test_01_negation_preservation_ptbr(self) -> None:
        neg_sentence = "Até o momento não recebi o produto e não recomendo essa loja!"
        cleaned = clean_pt_text(neg_sentence, preserve_negations=True)
        self.assertIn("nao_recebi", cleaned)
        self.assertIn("nao_recomendo", cleaned)
        self.assertLess(compute_lexicon_score(cleaned), -2.0)

        pos_sentence = "Produto excelente, entrega super rápida, chegou antes do prazo!"
        cleaned_pos = clean_pt_text(pos_sentence, preserve_negations=True)
        self.assertGreater(compute_lexicon_score(cleaned_pos), 3.0)

    def test_02_aspect_root_cause_detection(self) -> None:
        txt1 = clean_pt_text("Produto chegou quebrado e com defeito, quero reembolso!")
        asps1 = detect_aspects(txt1)
        self.assertIn("Defeito / Avaria", asps1)
        self.assertIn("Atendimento / Estorno / Troca", asps1)

        txt2 = clean_pt_text("Atrasou 15 dias nos correios e ainda faltou uma peça.")
        asps2 = detect_aspects(txt2)
        self.assertIn("Atraso / Entrega", asps2)
        self.assertIn("Pedido Incompleto / Faltando", asps2)

    def test_03_chronological_split_zero_overlap(self) -> None:
        dates = pd.date_range("2017-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "order_purchase_timestamp": dates,
            "texto_completo": ["produto bom"] * 100,
        })
        tr, va, te = chronological_split(df, 0.70, 0.15)
        self.assertEqual(len(tr) + len(va) + len(te), 100)
        self.assertLess(tr["order_purchase_timestamp"].max(), va["order_purchase_timestamp"].min())
        self.assertLess(va["order_purchase_timestamp"].max(), te["order_purchase_timestamp"].min())

    def test_04_rag_copilot_grounded_response(self) -> None:
        kb = pd.DataFrame({
            "texto_completo": [
                "Não recebi o produto, entrega atrasou muito nos correios.",
                "Produto veio quebrado e danificado na embalagem.",
                "Chegou antes do prazo, excelente qualidade, super recomendo!",
                "Faltou uma unidade do pedido, veio incompleto.",
                "Cor diferente da foto do anúncio, propaganda enganosa.",
                "Amei a compra, perfeito e bem embalado.",
            ],
            "review_score": [1, 1, 5, 2, 1, 5],
            "categoria_principal": ["informatica", "beleza_saude", "casa", "esporte", "moda", "casa"],
            "atraso_entrega_dias": [12.0, -2.0, -5.0, 0.0, -1.0, -4.0],
        })
        kb = enrich_text_signals(kb)
        vec = build_text_vectorizer()
        # Ajusta min_df=1 apenas no mini-teste sintético
        vec.transformer_list[0][1].set_params(min_df=1)
        vec.transformer_list[1][1].set_params(min_df=1)
        X_kb = vec.fit_transform(kb["texto_limpo"])

        rag = CustomerSupportRAG(vec, kb, X_kb)
        out = rag.assist(
            review_text="Atrasou 10 dias e até hoje não recebi meu pedido!",
            order_context={"categoria_principal": "informatica", "atraso_entrega_dias": 10.0, "valor_total": 350.0},
            pred_label="Negativo",
            prob_detrator=0.94,
            top_k=3,
        )
        self.assertIn("ALTA", out["prioridade_sac"])
        self.assertIn("Atraso / Entrega", out["aspectos_detectados"])
        self.assertEqual(len(out["casos_similares"]), 3)
        self.assertGreater(out["casos_similares"][0]["similarity"], 0.1)

    def test_05_multiclass_metrics(self) -> None:
        y_true = np.array(["Negativo", "Negativo", "Neutro", "Positivo", "Positivo"])
        y_pred = np.array(["Negativo", "Negativo", "Neutro", "Positivo", "Positivo"])
        proba = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.1, 0.80, 0.10],
            [0.05, 0.05, 0.90],
            [0.02, 0.08, 0.90],
        ])
        m = evaluate_multiclass(y_true, y_pred, proba)
        self.assertAlmostEqual(m["f1_macro"], 1.0)
        self.assertAlmostEqual(m["detrator_pr_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
