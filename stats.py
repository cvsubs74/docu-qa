import sqlite3


class Stats:
    def __init__(self, dbname):
        self.dbname = dbname
        self._initialize_counter_store()
        self._initialize_review_store()
        self._initialize_file_size_store()

    def _initialize_counter_store(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS doc_counters
                (id INTEGER PRIMARY KEY, count INTEGER)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS query_counters
                    (id INTEGER PRIMARY KEY, count INTEGER)''')
        connection.commit()
        connection.close()

    def _initialize_review_store(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        # Create the reviews table if it doesn't exist
        cursor.execute(f"CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, rating INTEGER, comment TEXT)")

    def _initialize_file_size_store(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS file_sizes
                (id INTEGER PRIMARY KEY, size INTEGER)''')
        connection.commit()
        connection.close()

    def add_document_count(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT count FROM doc_counters WHERE id = 1')
        count = cursor.fetchone()
        if count is None:
            cursor.execute('INSERT INTO doc_counters (id, count) VALUES (1, ?)', (1,))
        else:
            cursor.execute('UPDATE doc_counters SET count = ? WHERE id = 1', (count[0] + 1,))
        connection.commit()
        connection.close()

    def add_file_size(self, size):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT size FROM file_sizes WHERE id = 1')
        total_size = cursor.fetchone()
        if total_size is None:
            cursor.execute('INSERT INTO file_sizes (id, size) VALUES (1, ?)', (size,))
        else:
            cursor.execute('UPDATE file_sizes SET size = ? WHERE id = 1', (total_size[0] + size,))
        connection.commit()
        connection.close()

    def add_query_count(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT count FROM query_counters WHERE id = 1')
        count = cursor.fetchone()
        if count is None:
            cursor.execute('INSERT INTO query_counters (id, count) VALUES (1, ?)', (1,))
        else:
            cursor.execute('UPDATE query_counters SET count = ? WHERE id = 1', (count[0] + 1,))
        connection.commit()
        connection.close()

    def add_review(self, rating, comment):
        """Add a review to the database."""
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute(f"INSERT INTO reviews (rating, comment) VALUES (?, ?)", (rating, comment))
        connection.commit()

    def get_document_count(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT count FROM doc_counters WHERE id = 1')
        count = cursor.fetchone()
        connection.close()
        return count[0] if count is not None else 0

    def get_total_file_size(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT size FROM file_sizes WHERE id = 1')
        total_size = cursor.fetchone()
        connection.close()
        return total_size[0] if total_size is not None else 0

    def get_query_count(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute('SELECT count FROM query_counters WHERE id = 1')
        count = cursor.fetchone()
        connection.close()
        return count[0] if count is not None else 0

    def get_review_stats(self):
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*), AVG(rating) FROM reviews")
        result = cursor.fetchone()
        num_reviews = result[0]
        avg_rating = round(result[1], 1) if result[1] is not None else 0.0
        return num_reviews, avg_rating

    def get_reviews(self):
        """Get all the reviews from the database."""
        connection = sqlite3.connect(self.dbname)
        cursor = connection.cursor()
        cursor.execute("SELECT rating, comment FROM reviews WHERE comment != '' ORDER BY rating DESC LIMIT 10")
        return cursor.fetchall()




