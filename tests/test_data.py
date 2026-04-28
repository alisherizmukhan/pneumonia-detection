import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import get_dataloaders


def _create_dummy_dataset(root):
    """Create a minimal dummy dataset for testing."""
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            folder = Path(root) / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img = Image.fromarray(
                    (255 * __import__("numpy").random.rand(64, 64, 3)).astype("uint8")
                )
                img.save(folder / f"img_{i}.jpeg")


class TestDataLoader(unittest.TestCase):
    """Test that dataloaders load correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _create_dummy_dataset(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_dataloaders_return_three(self):
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=self.tmpdir, batch_size=2, image_size=224, num_workers=0
        )
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_dataloader_batch_shape(self):
        train_loader, _, _ = get_dataloaders(
            data_dir=self.tmpdir, batch_size=2, image_size=224, num_workers=0
        )
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[1:], (3, 224, 224))
        self.assertEqual(images.shape[0], 2)

    def test_dataloader_labels_binary(self):
        _, _, test_loader = get_dataloaders(
            data_dir=self.tmpdir, batch_size=4, image_size=224, num_workers=0
        )
        _, labels = next(iter(test_loader))
        unique = labels.unique().tolist()
        for label in unique:
            self.assertIn(label, [0, 1])


if __name__ == "__main__":
    unittest.main()
