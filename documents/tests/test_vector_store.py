from django.test import TestCase
from documents.models import Document, DocumentChunk
from documents.services.vector_store import ORMVectorStore, SearchResult


class ORMVectorStoreTests(TestCase):
    def setUp(self):
        self.doc1 = Document.objects.create(
            title="Doc1.pdf",
            file_type="pdf",
            status=Document.STATUS_READY,
        )
        self.doc2 = Document.objects.create(
            title="Doc2.txt",
            file_type="txt",
            status=Document.STATUS_READY,
        )
        self.doc_pending = Document.objects.create(
            title="DocPending.pdf",
            file_type="pdf",
            status=Document.STATUS_PENDING,
        )

        # Create chunks with known directional unit vectors (3 dimensions for testing)
        # Vector along X axis
        self.chunk1 = DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=0,
            page_number=1,
            content="Content along X axis",
            embedding=[1.0, 0.0, 0.0],
        )
        # Vector along Y axis
        self.chunk2 = DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=1,
            page_number=2,
            content="Content along Y axis",
            embedding=[0.0, 1.0, 0.0],
        )
        # Vector close to X axis (e.g. [0.9, 0.1, 0.0])
        self.chunk3 = DocumentChunk.objects.create(
            document=self.doc2,
            chunk_index=0,
            page_number=None,
            content="Content near X axis",
            embedding=[0.9, 0.1, 0.0],
        )
        # Chunk from pending document (must be excluded from results)
        self.chunk_pending = DocumentChunk.objects.create(
            document=self.doc_pending,
            chunk_index=0,
            page_number=1,
            content="Should not appear",
            embedding=[1.0, 0.0, 0.0],
        )

        self.store = ORMVectorStore()

    def test_search_ranks_closest_vector_first(self):
        # Query vector along X axis
        query_vector = [1.0, 0.0, 0.0]
        results = self.store.search(query_vector=query_vector, top_k=5)

        self.assertGreaterEqual(len(results), 2)
        # Top result should be chunk1 (exact match, similarity 1.0)
        self.assertEqual(results[0].chunk_id, self.chunk1.id)
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=4)

        # Second result should be chunk3 (near X axis)
        self.assertEqual(results[1].chunk_id, self.chunk3.id)
        self.assertGreater(results[1].similarity_score, 0.8)

        # Pending document chunk must not be in results
        chunk_ids = [r.chunk_id for r in results]
        self.assertNotIn(self.chunk_pending.id, chunk_ids)

    def test_search_filter_by_document_id(self):
        query_vector = [1.0, 0.0, 0.0]
        results = self.store.search(
            query_vector=query_vector,
            top_k=5,
            document_id=self.doc2.id,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, self.chunk3.id)

    def test_search_min_similarity_threshold(self):
        query_vector = [1.0, 0.0, 0.0]
        # chunk2 is orthogonal (dot product 0.0), so with threshold 0.5 it must be excluded
        results = self.store.search(
            query_vector=query_vector,
            top_k=5,
            min_similarity=0.5,
        )
        chunk_ids = [r.chunk_id for r in results]
        self.assertIn(self.chunk1.id, chunk_ids)
        self.assertIn(self.chunk3.id, chunk_ids)
        self.assertNotIn(self.chunk2.id, chunk_ids)

    def test_empty_query_returns_empty_list(self):
        self.assertEqual(self.store.search([]), [])
