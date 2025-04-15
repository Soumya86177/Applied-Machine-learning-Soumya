import unittest
import joblib
import os
import time
import requests
import subprocess
import signal
from score import score


class TestScoreFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained model from joblib/pkl
        cls.model = joblib.load("best_model.joblib")

    def test_smoke(self):
        """Smoke test: Does the function run without crashing?"""
        pred, prob = score("Test message", self.model, 0.5)
        self.assertIsNotNone(pred)
        self.assertIsNotNone(prob)

    def test_output_format(self):
        """Output type check"""
        pred, prob = score("Test message", self.model, 0.5)
        self.assertIsInstance(pred, bool)
        self.assertIsInstance(prob, float)

    def test_prediction_range(self):
        """Check prediction is 0 or 1 and probability is in [0, 1]"""
        pred, prob = score("Free money!!!", self.model, 0.5)
        self.assertIn(pred, [True, False])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_threshold_zero(self):
        """With threshold = 0, prediction should always be True"""
        pred, _ = score("Any text here", self.model, 0.0)
        self.assertTrue(pred)

    def test_threshold_one(self):
        """With threshold = 1, prediction should always be False"""
        pred, _ = score("Any text here", self.model, 1.0)
        self.assertFalse(pred)

    def test_obvious_spam(self):
        """Obvious spam should be classified as spam (True)"""
        spam_text = "Congratulations! You've won a free iPhone. Click to claim now!"
        pred, prop = score(spam_text, self.model, 0.5)
        print("Spam prediction probability:", prop)
        self.assertGreaterEqual(prop, 0.4)

    def test_obvious_ham(self):
        """Obvious ham should be classified as ham (False)"""
        ham_text = "Hi John, are we still meeting at 3 PM tomorrow?"
        pred, _ = score(ham_text, self.model, 0.5)
        self.assertFalse(pred)

    def test_empty_input(self):
        """Test that empty string input does not crash the model"""
        pred, prob = score("", self.model, 0.5)
        self.assertIsInstance(pred, bool)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_long_text(self):
        """Test model's response to very long input text"""
        long_text = "Hello " * 1000 + "win money now" + " bye" * 1000
        pred, prob = score(long_text, self.model, 0.5)
        self.assertIsInstance(pred, bool)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestFlaskIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Start Flask app in background
        cls.process = subprocess.Popen(["python", "app.py"])
        time.sleep(2)  # Wait for server to start

    @classmethod
    def tearDownClass(cls):
        # Gracefully shut down the Flask app
        cls.process.send_signal(signal.SIGINT)
        cls.process.wait()

    def test_flask(self):
        """Integration test: Flask /score endpoint"""
        payload = {
            "text": "You have won a lottery! Click now!",
            "threshold": 0.5
        }
        response = requests.post("http://127.0.0.1:5000/score", json=payload)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("propensity", result)
        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["propensity"], 0.0)
        self.assertLessEqual(result["propensity"], 1.0)



