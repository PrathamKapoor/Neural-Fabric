import csv
import tempfile
import unittest
from pathlib import Path
from neural_fabric.csv_data import fit_preprocessing, partition, read_csv
from neural_fabric.csv_workflow import train_csv, load_csv_model, predict_csv, evaluate_csv


class CsvWorkflowTest(unittest.TestCase):
    def test_training_reload_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "target"])
                for i in range(30):
                    writer.writerow([i, i % 3, "a" if i < 15 else "b"])
            names, rows, labels = read_csv(path, target="target")
            train_indices, _, test_indices = partition(labels, 42)
            self.assertFalse(set(train_indices) & set(test_indices))
            expected = fit_preprocessing([rows[i] for i in train_indices])
            model = str(Path(directory) / "model.pt")
            train_csv(path, "target", model, epochs=2)
            _, saved = load_csv_model(model)
            self.assertEqual(saved["preprocessing"], expected)
            self.assertEqual(len(predict_csv(model, path)), 30)
            self.assertEqual(evaluate_csv(model, path)["examples"], len(test_indices))
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same training dataset"):
                evaluate_csv(model, path)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for contents in ("x,x,target\n1,2,a\n", "x,target\nNaN,a\n", "x,target\nword,a\n"):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_csv(path, target="target")
