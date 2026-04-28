import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_model, BaselineCNN, ResNet18Model


class TestBaselineCNN(unittest.TestCase):

    def test_forward(self):
        model = BaselineCNN()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_output_is_logit(self):
        model = BaselineCNN()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1))

    def test_all_params_trainable(self):
        model = BaselineCNN()
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, f"{name} should be trainable")


class TestResNet18(unittest.TestCase):

    def test_forward(self):
        model = ResNet18Model(freeze_backbone=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_frozen_backbone(self):
        model = ResNet18Model(freeze_backbone=True)
        for name, param in model.backbone.named_parameters():
            if "fc" not in name:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
        # fc should still be trainable
        for param in model.backbone.fc.parameters():
            self.assertTrue(param.requires_grad)

    def test_output_shape(self):
        model = ResNet18Model()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1))


class TestDenseNet121(unittest.TestCase):

    def test_forward(self):
        model = get_model("densenet121", freeze_backbone=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_frozen_backbone(self):
        model = get_model("densenet121", freeze_backbone=True)
        for name, param in model.features.named_parameters():
            self.assertFalse(param.requires_grad, f"{name} should be frozen")
        for param in model.classifier.parameters():
            self.assertTrue(param.requires_grad)

    def test_output_shape(self):
        model = get_model("densenet121")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1))


class TestModelFactory(unittest.TestCase):

    def test_get_model_baseline(self):
        model = get_model("baseline")
        self.assertIsInstance(model, BaselineCNN)

    def test_get_model_resnet18(self):
        model = get_model("resnet18")
        self.assertIsInstance(model, ResNet18Model)

    def test_get_model_densenet121(self):
        model = get_model("densenet121")
        self.assertEqual(model.classifier.out_features, 1)

    def test_get_model_invalid(self):
        with self.assertRaises(ValueError):
            get_model("invalid_model")


if __name__ == "__main__":
    unittest.main()
