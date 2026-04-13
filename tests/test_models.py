import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_model


class TestModels(unittest.TestCase):
    """Test DenseNet121 model."""

    def test_densenet121_forward(self):
        model = get_model(freeze_backbone=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_densenet121_frozen_backbone(self):
        model = get_model(freeze_backbone=True)
        for name, param in model.features.named_parameters():
            self.assertFalse(param.requires_grad, f"{name} should be frozen")
        # Classifier should still be trainable
        for param in model.classifier.parameters():
            self.assertTrue(param.requires_grad)

    def test_densenet121_output_range(self):
        model = get_model()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Output should be a raw logit (single value)
        self.assertEqual(out.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
