import unittest
from unittest.mock import AsyncMock, patch

from src.clients.decision_client import DecisionManager


class DecisionClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        DecisionManager._relation_cache.clear()

    async def test_grounding_numeric_answer(self):
        with patch.object(
            DecisionManager,
            "_post_systemone",
            new=AsyncMock(return_value={"answers": {"grounded": 0.91}}),
        ):
            result = await DecisionManager.classify_grounding("chunk", "statement")
        self.assertTrue(result["is_grounded"])
        self.assertEqual(result["decision"], "accept")
        self.assertAlmostEqual(result["p_grounded"], 0.91)

    async def test_grounding_dict_answer_rejects(self):
        with patch.object(
            DecisionManager,
            "_post_systemone",
            new=AsyncMock(return_value={"answers": {"grounded": {"noul": 0.10}}}),
        ):
            result = await DecisionManager.classify_grounding("chunk", "statement")
        self.assertFalse(result["is_grounded"])
        self.assertEqual(result["decision"], "reject")

    async def test_grounding_batch_skips_empty_statements(self):
        mock_post = AsyncMock(return_value={"answers": {"q_0": 0.90}})
        with patch.object(DecisionManager, "_post_systemone", new=mock_post):
            results = await DecisionManager.classify_grounding_batch(
                "chunk", ["grounded statement", "   "]
            )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["decision"], "accept")
        self.assertEqual(results[1]["decision"], "reject")
        self.assertEqual(results[1]["p_grounded"], 0.0)
        self.assertEqual(set(mock_post.await_args.args[0]["questions"]), {"q_0"})

    async def test_relation_guard_and_cache(self):
        mock_post = AsyncMock(
            return_value={
                "answers": {
                    "relation": {
                        "choice": "none",
                        "probabilities": {"none": 0.95},
                        "confidence": 0.95,
                    }
                }
            }
        )
        with patch.object(DecisionManager, "_post_systemone", new=mock_post):
            first = await DecisionManager.classify_relation("a", "b")
            second = await DecisionManager.classify_relation("a ", " b")
        self.assertTrue(first["is_uncoupled"])
        self.assertIs(first, second)
        self.assertEqual(mock_post.await_count, 1)


if __name__ == "__main__":
    unittest.main()
