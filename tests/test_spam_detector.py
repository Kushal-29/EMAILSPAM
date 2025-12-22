import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import EmailPreprocessor
from src.model_training import SpamClassifier
import numpy as np

class TestSpamDetector(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = EmailPreprocessor()
        self.classifier = SpamClassifier('naive_bayes')
    
    def test_clean_text(self):
        test_text = "HELLO World! This is a test 123."
        cleaned = self.preprocessor.clean_text(test_text)
        self.assertEqual(cleaned, "hello world this is a test")
    
    def test_tokenize_and_stem(self):
        test_text = "running runs ran"
        processed = self.preprocessor.tokenize_and_stem(test_text)
        # Should stem to 'run' variations
        self.assertIn('run', processed)
    
    def test_classifier_initialization(self):
        self.assertEqual(self.classifier.model_type, 'naive_bayes')
        self.assertIsNotNone(self.classifier.model)
    
    def test_svm_classifier(self):
        svm_classifier = SpamClassifier('svm')
        self.assertEqual(svm_classifier.model_type, 'svm')

if __name__ == '__main__':
    unittest.main()