import unittest
import shutil
from pathlib import Path
from src.utils import generate_synthetic_data, measure_rss_mb

class TestIOBenchmark(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = Path("tests/temp_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.test_dir / "test_synthetic.csv"
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_generate_synthetic_data_creates_file(self):
        generate_synthetic_data(self.csv_path, n_rows=100)
        self.assertTrue(self.csv_path.exists())
        self.assertGreater(self.csv_path.stat().st_size, 0)
        
    def test_measure_rss_returns_float(self):
        rss = measure_rss_mb()
        self.assertIsInstance(rss, float)
        self.assertGreater(rss, 0)

if __name__ == '__main__':
    unittest.main()
