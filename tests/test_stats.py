import os
import unittest
import tempfile

from stats import Stats


class TestStats(unittest.TestCase):

    def setUp(self):
        # Use a temporary file as the database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.stats = Stats(self.db_path)

    def tearDown(self):
        # Close the database and delete the temporary file
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_add_document_count(self):
        # Test that the document count is incremented
        initial_count = self.stats.get_document_count()
        self.stats.add_document_count()
        self.assertEqual(self.stats.get_document_count(), initial_count + 1)

    def test_add_file_size(self):
        # Test that the file size is added correctly
        initial_size = self.stats.get_total_file_size()
        self.stats.add_file_size(100)
        self.assertEqual(self.stats.get_total_file_size(), initial_size + 100)

    def test_add_query_count(self):
        # Test that the query count is incremented
        initial_count = self.stats.get_query_count()
        self.stats.add_query_count()
        self.assertEqual(self.stats.get_query_count(), initial_count + 1)

    def test_add_review(self):
        # Test that a review is added correctly
        initial_num_reviews, initial_avg_rating = self.stats.get_review_stats()
        self.stats.add_review(4, "This app is great!")
        self.assertEqual(self.stats.get_review_stats()[0], initial_num_reviews + 1)

    def test_get_document_count(self):
        # Test that the document count is retrieved correctly
        self.stats.add_document_count()
        self.assertEqual(self.stats.get_document_count(), 1)

    def test_get_total_file_size(self):
        # Test that the total file size is retrieved correctly
        self.stats.add_file_size(100)
        self.assertEqual(self.stats.get_total_file_size(), 100)

    def test_get_query_count(self):
        # Test that the query count is retrieved correctly
        self.stats.add_query_count()
        self.assertEqual(self.stats.get_query_count(), 1)

    def test_get_review_stats(self):
        # Test that the review stats are retrieved correctly
        self.stats.add_review(4, "This app is great!")
        self.stats.add_review(5, "Awesome app!")
        num_reviews, avg_rating = self.stats.get_review_stats()
        self.assertEqual(num_reviews, 2)
        self.assertEqual(avg_rating, 4.5)

    def test_get_reviews(self):
        # Test that the reviews are retrieved correctly
        self.stats.add_review(4, "This app is great!")
        self.stats.add_review(5, "Awesome app!")
        reviews = self.stats.get_reviews()
        self.assertEqual(len(reviews), 2)
        self.assertEqual(reviews[0][0], 5)
        self.assertEqual(reviews[1][1], "This app is great!")
