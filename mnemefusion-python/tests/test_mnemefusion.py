"""
Unit tests for MnemeFusion Python bindings
"""

import os
import tempfile
import pytest


def random_embedding(dim=384):
    """Generate a random embedding for testing"""
    import random
    return [random.random() for _ in range(dim)]


class TestMemoryBasics:
    """Test basic memory operations"""

    def test_create_and_open_database(self):
        """Test creating/opening a database"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            memory = mnemefusion.Memory(db_path)
            assert memory.count() == 0
            memory.close()

    def test_context_manager(self):
        """Test using Memory as a context manager"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                assert memory.count() == 0

    def test_custom_config(self):
        """Test opening database with custom config"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            memory = mnemefusion.Memory(db_path, config={
                "embedding_dim": 512,
                "entity_extraction_enabled": False,
            })
            assert memory.count() == 0
            memory.close()

    def test_add_memory(self):
        """Test adding a memory"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                memory_id = memory.add(
                    "Test content",
                    random_embedding(),
                )
                assert isinstance(memory_id, str)
                assert len(memory_id) > 0
                assert memory.count() == 1

    def test_add_memory_with_metadata(self):
        """Test adding a memory with metadata"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                memory_id = memory.add(
                    "Test content",
                    random_embedding(),
                    metadata={"key": "value", "type": "test"}
                )
                assert memory.count() == 1

                # Retrieve and verify
                result = memory.get(memory_id)
                assert result is not None
                assert result["content"] == "Test content"
                assert result["metadata"]["key"] == "value"

    def test_add_memory_with_timestamp(self):
        """Test adding a memory with custom timestamp"""
        import mnemefusion
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                ts = time.time()
                memory_id = memory.add(
                    "Test content",
                    random_embedding(),
                    timestamp=ts
                )

                result = memory.get(memory_id)
                assert result is not None
                assert abs(result["created_at"] - ts) < 1.0

    def test_get_memory(self):
        """Test retrieving a memory by ID"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                embedding = random_embedding()
                memory_id = memory.add("Test content", embedding)

                result = memory.get(memory_id)
                assert result is not None
                assert result["id"] == memory_id
                assert result["content"] == "Test content"
                assert len(result["embedding"]) == 384
                assert isinstance(result["created_at"], float)

    def test_get_nonexistent_memory(self):
        """Test getting a memory that doesn't exist"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                # Use a valid ID format but one that doesn't exist
                result = memory.get("00000000-0000-0000-0000-000000000001")
                assert result is None

    def test_delete_memory(self):
        """Test deleting a memory"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                memory_id = memory.add("Test content", random_embedding())
                assert memory.count() == 1

                deleted = memory.delete(memory_id)
                assert deleted is True
                assert memory.count() == 0

                # Verify it's gone
                result = memory.get(memory_id)
                assert result is None

    def test_delete_nonexistent_memory(self):
        """Test deleting a memory that doesn't exist"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                deleted = memory.delete("00000000-0000-0000-0000-000000000001")
                assert deleted is False


class TestSearch:
    """Test search functionality"""

    def test_search_empty_database(self):
        """Test searching an empty database"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                results = memory.search(random_embedding(), top_k=10)
                assert len(results) == 0

    def test_search_returns_results(self):
        """Test that search returns results"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                # Add some memories
                for i in range(5):
                    memory.add(f"Content {i}", random_embedding())

                # Search
                results = memory.search(random_embedding(), top_k=3)
                assert len(results) == 3

                # Each result should be a tuple of (memory_dict, score)
                for mem_dict, score in results:
                    assert isinstance(mem_dict, dict)
                    assert "id" in mem_dict
                    assert "content" in mem_dict
                    assert isinstance(score, float)
                    assert 0.0 <= score <= 1.0


class TestQuery:
    """Test intelligent query functionality"""

    def test_query_empty_database(self):
        """Test querying an empty database"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                intent, results = memory.query(
                    "Test query",
                    random_embedding(),
                    limit=10
                )

                assert isinstance(intent, dict)
                assert "intent" in intent
                assert "confidence" in intent
                assert len(results) == 0

    def test_query_returns_intent_and_results(self):
        """Test that query returns intent classification and results"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                # Add some memories
                for i in range(5):
                    memory.add(f"Content {i}", random_embedding())

                # Query
                intent, results = memory.query(
                    "Why did this happen?",
                    random_embedding(),
                    limit=3
                )

                # Check intent
                assert isinstance(intent, dict)
                assert "intent" in intent
                assert "confidence" in intent
                assert isinstance(intent["intent"], str)
                assert isinstance(intent["confidence"], float)

                # Check results
                assert len(results) <= 3
                for mem_dict, scores_dict in results:
                    assert isinstance(mem_dict, dict)
                    assert "id" in mem_dict
                    assert "content" in mem_dict

                    assert isinstance(scores_dict, dict)
                    assert "semantic_score" in scores_dict
                    assert "temporal_score" in scores_dict
                    assert "causal_score" in scores_dict
                    assert "entity_score" in scores_dict
                    assert "fused_score" in scores_dict


class TestCausalGraph:
    """Test causal graph operations"""

    def test_add_causal_link(self):
        """Test adding a causal link"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                cause_id = memory.add("Cause", random_embedding())
                effect_id = memory.add("Effect", random_embedding())

                memory.add_causal_link(
                    cause_id,
                    effect_id,
                    0.9,
                    "Test causal relationship"
                )

    def test_get_causes(self):
        """Test getting causes of a memory"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                cause_id = memory.add("Cause", random_embedding())
                effect_id = memory.add("Effect", random_embedding())

                memory.add_causal_link(cause_id, effect_id, 0.9, "Test")

                # Get causes of the effect
                causes = memory.get_causes(effect_id, max_hops=3)
                assert isinstance(causes, list)
                # Should find at least one path
                assert len(causes) >= 1
                # Each path is a list of memory IDs
                for path in causes:
                    assert isinstance(path, list)
                    for mem_id in path:
                        assert isinstance(mem_id, str)

    def test_get_effects(self):
        """Test getting effects of a memory"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                cause_id = memory.add("Cause", random_embedding())
                effect_id = memory.add("Effect", random_embedding())

                memory.add_causal_link(cause_id, effect_id, 0.9, "Test")

                # Get effects of the cause
                effects = memory.get_effects(cause_id, max_hops=3)
                assert isinstance(effects, list)
                assert len(effects) >= 1


class TestEntities:
    """Test entity operations"""

    def test_list_entities(self):
        """Test listing entities"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                # Add memories that might contain entities
                memory.add("Alice met Bob", random_embedding())
                memory.add("Bob talked to Charlie", random_embedding())

                # List entities
                entities = memory.list_entities()
                assert isinstance(entities, list)

                # Each entity should be a dict
                for entity in entities:
                    assert isinstance(entity, dict)
                    assert "id" in entity
                    assert "name" in entity
                    assert "mention_count" in entity
                    assert isinstance(entity["mention_count"], int)


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_memory_id(self):
        """Test that invalid memory IDs raise errors"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path) as memory:
                with pytest.raises(ValueError):
                    memory.get("invalid-id")

    def test_operations_after_close(self):
        """Test that operations after close raise errors"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            memory = mnemefusion.Memory(db_path)
            memory.close()

            with pytest.raises(RuntimeError):
                memory.add("Test", random_embedding())

    def test_wrong_embedding_dimension(self):
        """Test that wrong embedding dimension raises error"""
        import mnemefusion

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.mfdb")
            with mnemefusion.Memory(db_path, config={"embedding_dim": 384}) as memory:
                # Try to add with wrong dimension
                with pytest.raises(ValueError):
                    memory.add("Test", [0.1] * 512)  # Wrong dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
