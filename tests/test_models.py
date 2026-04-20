import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_model, BaselineCNN


class TestBaselineCNN(unittest.TestCase):

    def test_forward(self):
        model = get_model("baseline")
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_output_is_logit(self):
        model = get_model("baseline")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1))

    def test_all_params_trainable(self):
        model = get_model("baseline")
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, f"{name} should be trainable")


class TestResNet18(unittest.TestCase):

    def test_forward(self):
        model = get_model("resnet18")
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_frozen_backbone(self):
        model = get_model("resnet18", freeze_backbone=True)
        for name, param in model.named_parameters():
            if "fc" not in name:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
        for param in model.fc.parameters():
            self.assertTrue(param.requires_grad)

    def test_output_shape(self):
        model = get_model("resnet18")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1))


class TestDenseNet121(unittest.TestCase):

    def test_forward(self):
        model = get_model("densenet121")
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


class TestGetModelDispatch(unittest.TestCase):

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_model("unknown_arch")

    def test_default_is_densenet121(self):
        model = get_model()
        from torchvision.models import DenseNet
        self.assertIsInstance(model, DenseNet)


if __name__ == "__main__":
    unittest.main()
