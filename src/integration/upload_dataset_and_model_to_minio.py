# src/upload_to_minio.py
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

from dotenv import load_dotenv

# ------------------------------------------------------------
# CONFIG: adjust these two paths to match your machine
# ------------------------------------------------------------
SMART_ENERGY_REPO = Path(r"C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea")
MINIO_CLIENT_REPO = Path(r"C:\Users\gl8304\Documents\Projekti\IJS\minio-ai-storage")

# Make minio_humaine_client importable without pip install -e .
sys.path.insert(0, str(MINIO_CLIENT_REPO))

from minio_humaine_client.auth import HumaineAuth  # noqa: E402


def assert_file_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{label} must be a file, got directory: {p}")


def upload_file(
    auth: HumaineAuth,
    bucket: str,
    local_path: Path,
    object_key: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Upload file to a selected bucket and optionally attach metadata.

    Note: metadata update may fail (server-side 500). We treat metadata as best-effort.
    """
    auth.select_bucket(bucket)
    print(f"📦 Uploading to bucket='{bucket}' key='{object_key}' from '{local_path}'")
    auth.upload_object(object_key, str(local_path))

    if metadata:
        try:
            auth.update_metadata(object_key, metadata)
            print(f"🏷️  Metadata updated for '{object_key}'")
        except Exception as e:
            print(f"⚠️  Metadata update failed for '{object_key}': {e}")
            print("   -> continuing without metadata (manifest will link artifacts)")


def main():
    # ------------------------------------------------------------
    # Load env vars from smart-energy-ea/.env
    # ------------------------------------------------------------
    load_dotenv(dotenv_path=SMART_ENERGY_REPO / ".env")

    base_url = os.getenv("HUMAINE_API_BASE_URL")
    username = os.getenv("HUMAINE_API_USERNAME")
    password = os.getenv("HUMAINE_API_PASSWORD")

    if not base_url or not username or not password:
        raise RuntimeError(
            "Missing env vars. Set in smart-energy-ea/.env:\n"
            "  HUMAINE_API_BASE_URL\n"
            "  HUMAINE_API_USERNAME\n"
            "  HUMAINE_API_PASSWORD\n"
        )

    # ------------------------------------------------------------
    # Local artifact paths (from your smart-energy-ea repo)
    # ------------------------------------------------------------
    dataset_path = SMART_ENERGY_REPO / "data" / "simulation_security_labels_n-1.csv"
    model_path = SMART_ENERGY_REPO / "models" / "random_forest_model.pkl"

    assert_file_exists(dataset_path, "Dataset")
    assert_file_exists(model_path, "Model")

    # ------------------------------------------------------------
    # Remote object keys (where in MinIO)
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_key = f"training_datasets/simulation_security_labels_n-1_{timestamp}.csv"
    model_key = f"models/random_forest_model_{timestamp}.pkl"

    # ------------------------------------------------------------
    # Connect + login
    # ------------------------------------------------------------
    print("🔐 Logging in to HumAIne API...")
    auth = HumaineAuth(base_url)
    auth.login(username, password)
    auth.list_buckets()  # important: select_bucket validates against cached list
    print("✅ Login OK")

    # ------------------------------------------------------------
    # Upload dataset
    # ------------------------------------------------------------
    dataset_metadata = {
        "type": "training_dataset",
        "source_repo": "smart-energy-ea",
        "filename": dataset_path.name,
        "created_at": timestamp,
        "notes": "Dataset used for AL/RF security classification training.",
    }
    upload_file(
        auth=auth,
        bucket="smart-energy-data",
        local_path=dataset_path,
        object_key=dataset_key,
        metadata=dataset_metadata,
    )

    # ------------------------------------------------------------
    # Upload model
    # ------------------------------------------------------------
    model_metadata = {
        "type": "model",
        "model_type": "random_forest",
        "source_repo": "smart-energy-ea",
        "filename": model_path.name,
        "created_at": timestamp,
        "trained_on_dataset_key": dataset_key,
        "notes": "RF classifier uploaded for pilot integration.",
    }
    upload_file(
        auth=auth,
        bucket="smart-energy-models",
        local_path=model_path,
        object_key=model_key,
        metadata=model_metadata,
    )

    # ------------------------------------------------------------
    # Upload manifest (links dataset + model, no results)
    # ------------------------------------------------------------
    manifest = {
        "timestamp": timestamp,
        "dataset": {"bucket": "smart-energy-data", "key": dataset_key},
        "model": {"bucket": "smart-energy-models", "key": model_key},
        "notes": "Metadata endpoint may return 500; this manifest links the artifacts.",
    }

    manifest_path = SMART_ENERGY_REPO / "tmp_minio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    auth.select_bucket("smart-energy-models")
    manifest_key = f"manifests/{timestamp}/manifest.json"
    print(f"📦 Uploading manifest: {manifest_path} -> {manifest_key}")
    auth.upload_object(manifest_key, str(manifest_path))

    print("\n✅ DONE")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
