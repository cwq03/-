"""Tests for the sentence similarity engine."""

from __future__ import annotations

import unittest

from nlp_similarity import SentenceSimilarityEngine
from nlp_similarity.ml_model import train_logistic_model


class SentenceSimilarityEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SentenceSimilarityEngine()

    def test_synonym_sentences_are_similar(self) -> None:
        result = self.engine.compare("我喜欢自然语言处理", "我对NLP很感兴趣")
        self.assertGreaterEqual(result.score, 0.55)

    def test_unrelated_sentences_are_not_similar(self) -> None:
        result = self.engine.compare("今天北京天气很好", "深度学习模型可以处理文本")
        self.assertLess(result.score, 0.42)

    def test_identical_sentences_have_high_score(self) -> None:
        result = self.engine.compare("语义相似度计算很重要", "语义相似度计算很重要")
        self.assertGreaterEqual(result.score, 0.9)

    def test_result_contains_explanations(self) -> None:
        result = self.engine.compare("老师讲解文本相似度", "教师介绍语义匹配")
        self.assertTrue(result.tokens_a)
        self.assertTrue(result.tokens_b)
        self.assertIn("老师", result.normalized_tokens_b)

    def test_negation_mismatch_lowers_score(self) -> None:
        positive = self.engine.compare("我喜欢这门课程", "我喜欢这门课程")
        negative = self.engine.compare("我喜欢这门课程", "我不喜欢这门课程")
        self.assertTrue(negative.negation_mismatch)
        self.assertLess(negative.score, positive.score)
        self.assertLess(negative.score, 0.70)

    def test_rank_candidates_returns_top_matches(self) -> None:
        results = self.engine.rank_candidates(
            "我对NLP很感兴趣",
            ["今天北京天气很好", "我喜欢自然语言处理", "医生正在看病"],
            top_k=2,
        )
        self.assertEqual(results[0].sentence_b, "我喜欢自然语言处理")
        self.assertEqual(len(results), 2)

    def test_supervised_model_can_predict(self) -> None:
        rows = [
            ("我喜欢自然语言处理", "我对NLP很感兴趣", 1),
            ("今天北京天气很好", "医生正在给病人看病", 0),
            ("这家餐厅味道不错", "这家店很好吃", 1),
            ("我喜欢这门课程", "我不喜欢这门课程", 0),
        ]
        model = train_logistic_model(rows, self.engine, epochs=20, learning_rate=0.05)
        result = self.engine.compare("我喜欢自然语言处理", "我对NLP很感兴趣")
        prediction = model.predict(result)
        self.assertGreaterEqual(prediction.probability, 0.0)
        self.assertLessEqual(prediction.probability, 1.0)


if __name__ == "__main__":
    unittest.main()
