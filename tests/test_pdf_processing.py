import unittest
from src.embed_and_store import read_and_chunk_pdf, store_chunks

class TestPDFProcessing(unittest.TestCase):
    def test_read_and_chunk_pdf(self):
        # Test with a sample PDF
        chunks = read_and_chunk_pdf("data/sample.pdf")
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

    def test_store_chunks(self):
        # Test storing chunks
        chunks = ["Test chunk 1", "Test chunk 2"]
        try:
            store_chunks(chunks)
            self.assertTrue(True)  # If no exception is raised, test passes
        except Exception as e:
            self.fail(f"store_chunks raised an exception: {e}")

if __name__ == '__main__':
    unittest.main() 