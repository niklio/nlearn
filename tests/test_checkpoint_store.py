import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nlearn import checkpoint_store


class FakeBucket:
    def __init__(self):
        self.files = {}

    def batch(self, bucket_id, add=None, copy=None, delete=None, token=None):
        assert bucket_id == "owner/checkpoints"
        for source, destination in add or []:
            self.files[destination] = (
                bytes(source) if isinstance(source, bytes) else Path(source).read_bytes()
            )
        for path in delete or []:
            self.files.pop(path, None)

    def download(self, bucket_id, files, raise_on_missing_files=False, token=None):
        assert bucket_id == "owner/checkpoints"
        for source, destination in files:
            if source not in self.files:
                if raise_on_missing_files:
                    raise type("EntryNotFoundError", (Exception,), {})(source)
                continue
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.files[source])


class CheckpointStoreTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fake = FakeBucket()
        self.env = patch.dict(
            os.environ,
            {"NLEARN_HF_BUCKET": "owner/checkpoints"},
        )
        self.functions = patch.object(
            checkpoint_store,
            "_hf_functions",
            return_value=(self.fake.batch, self.fake.download),
        )
        self.env.start()
        self.functions.start()
        os.environ.pop("NLEARN_HF_PREFIX", None)

    def tearDown(self):
        self.functions.stop()
        self.env.stop()
        self.tempdir.cleanup()

    def test_bucket_id_accepts_hf_uri_and_rejects_paths(self):
        self.assertEqual(
            checkpoint_store.normalize_bucket_id(
                "hf://buckets/owner/checkpoints/"
            ),
            "owner/checkpoints",
        )
        with self.assertRaises(ValueError):
            checkpoint_store.normalize_bucket_id("owner/checkpoints/extra")
        self.assertEqual(checkpoint_store._prefix(None), "runs/_default")

    def test_machine_local_config_can_reference_token_file(self):
        token_file = self.root / "hf.key"
        token_file.write_text("hf_test_token_value_1234567890\n")
        config = self.root / "hf.env"
        config.write_text(
            f"HF_TOKEN_FILE={token_file}\n"
            "NLEARN_HF_BUCKET=owner/checkpoints\n"
        )
        with patch.dict(
            os.environ, {"NLEARN_HF_CONFIG": str(config)}, clear=True
        ):
            self.assertEqual(checkpoint_store.bucket_id(), "owner/checkpoints")
            self.assertEqual(
                checkpoint_store._token(), "hf_test_token_value_1234567890"
            )

    def test_upload_retention_and_download_latest(self):
        run_dir = self.root / "hero"
        run_dir.mkdir()
        old = run_dir / "step_000100.pkl"
        latest = run_dir / "step_000200.pkl"
        old.write_bytes(b"old parameters")
        latest.write_bytes(b"latest parameters")

        self.assertTrue(checkpoint_store.upload_step_checkpoint(old, 100, "hero", [old]))
        self.assertTrue(
            checkpoint_store.upload_step_checkpoint(
                latest, 200, "hero", [latest], [old]
            )
        )

        self.assertNotIn("runs/hero/steps/step_000100.pkl", self.fake.files)
        self.assertEqual(
            self.fake.files["runs/hero/steps/step_000200.pkl"], b"latest parameters"
        )
        manifest = json.loads(self.fake.files["runs/hero/manifest.json"])
        self.assertEqual(manifest["latest_step"], 200)
        self.assertEqual(manifest["steps"], ["step_000200.pkl"])

        downloaded = checkpoint_store.download_inference_checkpoint(
            "hero", destination_dir=self.root / "download"
        )
        self.assertEqual(downloaded.name, "step_000200.pkl")
        self.assertEqual(downloaded.read_bytes(), b"latest parameters")

    def test_resume_upload_download_and_missing_resume(self):
        source = self.root / "source" / "resume.pkl"
        source.parent.mkdir()
        source.write_bytes(b"optimizer and params")
        self.assertTrue(checkpoint_store.upload_resume_checkpoint(source, 321, "hero"))

        target = self.root / "restored" / "resume.pkl"
        self.assertEqual(
            checkpoint_store.download_resume_checkpoint(target, "hero"), target
        )
        self.assertEqual(target.read_bytes(), b"optimizer and params")

        missing_type = type("EntryNotFoundError", (Exception,), {})
        with patch.object(
            checkpoint_store, "_download", side_effect=missing_type("missing")
        ):
            self.assertIsNone(
                checkpoint_store.download_resume_checkpoint(
                    self.root / "missing" / "resume.pkl", "new-run"
                )
            )

    def test_numbered_checkpoint_selector(self):
        self.fake.files["runs/hero/steps/step_000042.pkl"] = b"step 42"
        path = checkpoint_store.download_inference_checkpoint(
            "hero", "42", destination_dir=self.root
        )
        self.assertEqual(path.read_bytes(), b"step 42")


if __name__ == "__main__":
    unittest.main()
