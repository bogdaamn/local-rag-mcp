import numpy as np
import faiss

import rag.query as query


def test_vector_search_returns_ids_from_index_search(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return np.array([[0.1, 0.2]], dtype="float32")

    class FakeIndex:
        def search(self, q_emb, top_k):
            return np.array([[0.9, 0.5]]), np.array([[7, 3]])

    monkeypatch.setattr(query, "model", FakeModel())
    monkeypatch.setattr(query, "index", FakeIndex())
    monkeypatch.setattr(query, "chunks", [{"id": 7}, {"id": 3}])
    monkeypatch.setattr(faiss, "normalize_L2", lambda x: None)

    result = query._vector_search("cats", top_k=2)

    assert result == [7, 3]


def test_vector_search_returns_empty_list_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(query, "index", None)
    monkeypatch.setattr(query, "chunks", [])

    assert query._vector_search("cats", top_k=5) == []


def test_hybrid_search_task_fuses_per_variant_rankings():
    calls = []

    def fake_search_fn(variant, top_k):
        calls.append((variant, top_k))
        return {"cats": [1, 2, 3], "kittens": [2, 1, 4]}[variant]

    result = query._hybrid_search_task(fake_search_fn, ["cats", "kittens"], top_k=10, rrf_k=60)

    assert calls == [("cats", 10), ("kittens", 10)]
    assert result == [1, 2, 3, 4]
