import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from rag_engine import RAGEngine
from vector_store import VectorStore


class RAGEngineInitializationTests(unittest.TestCase):
    def test_init_handles_groq_client_errors(self):
        with patch("rag_engine.Groq", side_effect=TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")):
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False):
                engine = RAGEngine(VectorStore())

        self.assertIsNone(engine.client)


if __name__ == "__main__":
    unittest.main()
