from django.test import TestCase

from documents.models import Document, DocumentChunk
from documents.services.vector_store import ORMVectorStore

M = "test-embed"


def mk_chunk(doc, idx, page, text, emb, model=M, dim=None):
    return DocumentChunk.objects.create(
        document=doc, chunk_index=idx, page_number=page, content=text, embedding=emb,
        embedding_model=model, embedding_dim=len(emb) if dim is None else dim)


class ORMVectorStoreTests(TestCase):
    def setUp(self):
        self.d1 = Document.objects.create(title="One.pdf", file_type="pdf", status=Document.STATUS_READY)
        self.d2 = Document.objects.create(title="Two.txt", file_type="txt", status=Document.STATUS_READY)
        self.failed = Document.objects.create(title="Bad.pdf", file_type="pdf", status=Document.STATUS_FAILED)
        self.pending = Document.objects.create(title="Wait.pdf", file_type="pdf", status=Document.STATUS_PENDING)
        self.x = mk_chunk(self.d1, 0, 1, "x axis", [1.0, 0.0, 0.0])
        self.y = mk_chunk(self.d1, 1, 2, "y axis", [0.0, 1.0, 0.0])
        self.nx = mk_chunk(self.d2, 0, None, "near x", [0.9, 0.1, 0.0])
        mk_chunk(self.failed, 0, 1, "failed doc", [1.0, 0.0, 0.0])
        mk_chunk(self.pending, 0, 1, "pending doc", [1.0, 0.0, 0.0])
        self.store = ORMVectorStore()

    def test_cosine_ordering_and_exact_score(self):
        r = self.store.search([1.0, 0.0, 0.0], top_k=5)
        self.assertEqual([x.chunk_id for x in r[:2]], [self.x.id, self.nx.id])
        self.assertAlmostEqual(r[0].similarity_score, 1.0, places=6)
        self.assertAlmostEqual(r[1].similarity_score, 0.9 / (0.9 ** 2 + 0.1 ** 2) ** 0.5, places=6)

    def test_scale_invariance(self):
        a = self.store.search([1.0, 0.0, 0.0])[0].similarity_score
        b = self.store.search([42.0, 0.0, 0.0])[0].similarity_score
        self.assertAlmostEqual(a, b, places=6)

    def test_excludes_failed_and_pending_documents(self):
        titles = {x.document_title for x in self.store.search([1.0, 0.0, 0.0], top_k=10)}
        self.assertEqual(titles, {"One.pdf", "Two.txt"})

    def test_document_filter(self):
        r = self.store.search([1.0, 0.0, 0.0], top_k=10, document_id=self.d2.id)
        self.assertEqual([x.chunk_id for x in r], [self.nx.id])

    def test_all_documents_when_no_filter(self):
        self.assertEqual(len(self.store.search([1.0, 0.0, 0.0], top_k=10)), 3)

    def test_threshold(self):
        ids = [x.chunk_id for x in self.store.search([1.0, 0.0, 0.0], min_similarity=0.5)]
        self.assertIn(self.x.id, ids)
        self.assertNotIn(self.y.id, ids)

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.store.search([1.0, 0.0, 0.0], top_k=1)), 1)

    def test_invalid_queries_return_empty(self):
        for q in ([], None, [0.0, 0.0, 0.0], [float("nan"), 0, 0], [[1, 0, 0]]):
            self.assertEqual(self.store.search(q), [], msg=str(q))
        self.assertEqual(self.store.search([1, 0, 0], top_k=0), [])

    def test_dimension_mismatch_returns_empty_not_error(self):
        self.assertEqual(self.store.search([1.0, 0.0]), [])

    def test_model_mismatch_excluded(self):
        self.assertEqual(self.store.search([1.0, 0.0, 0.0], embedding_model="other-model"), [])

    def test_missing_embeddings_and_empty_collection(self):
        DocumentChunk.objects.all().delete()
        self.assertEqual(self.store.search([1.0, 0.0, 0.0]), [])
        mk_chunk(self.d1, 0, 1, "no vector", [], dim=0)
        self.assertEqual(self.store.search([1.0, 0.0, 0.0]), [])

    def test_zero_stored_vector_never_matches(self):
        DocumentChunk.objects.all().delete()
        mk_chunk(self.d1, 0, 1, "zero", [0.0, 0.0, 0.0])
        self.assertEqual(self.store.search([1.0, 0.0, 0.0], min_similarity=0.0), [])

    def test_count_incompatible(self):
        self.assertEqual(self.store.count_incompatible(M, 3), 0)
        mk_chunk(self.d1, 5, 3, "old", [1.0] * 4, model="retired-model")
        self.assertEqual(self.store.count_incompatible(M, 3), 1)
        self.assertEqual(self.store.count_incompatible(M, 3, document_id=self.d2.id), 0)
