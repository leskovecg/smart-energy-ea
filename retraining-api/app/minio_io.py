"""
HumAIne MinIO wrapper.
"""

from __future__ import annotations

from pathlib import Path
import os
import requests

from minio_humaine_client.auth import HumaineAuth
from minio_humaine_client.data import download_data


def get_humaine_auth() -> HumaineAuth:
    base_url = os.getenv("HUMAINE_API_BASE_URL")
    username = os.getenv("HUMAINE_API_USERNAME")
    password = os.getenv("HUMAINE_API_PASSWORD")

    if not base_url or not username or not password:
        raise RuntimeError(
            "Missing env vars: HUMAINE_API_BASE_URL, HUMAINE_API_USERNAME, HUMAINE_API_PASSWORD"
        )

    auth = HumaineAuth(base_url)
    auth.login(username, password)

    # FIX: nujno, da auth.buckets ni prazen (select_bucket preverja buckets)
    try:
        auth.list_buckets()
    except Exception:
        # če endpoint ne obstaja / client ne podpira, vsaj ne crash-aj tukaj
        pass

    return auth


def download_object_to_path(auth, bucket, key, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # bucket selection (če obstaja)
    if hasattr(auth, "list_buckets"):
        try:
            auth.list_buckets()
        except Exception:
            pass

    if hasattr(auth, "select_bucket"):
        try:
            auth.select_bucket(bucket)
        except Exception as e:
            raise RuntimeError(f"Bucket select failed for '{bucket}': {e}")

    headers = auth.headers() if hasattr(auth, "headers") else {}

    # 1) tvoja trenutna (stara) varianta – verjetno NE dela
    candidates = [
        (f"{auth.base_url}/data/download", {"key": key}),
        # 2) bolj logična varianta za HumAIne: main_ops download s query parametri
        (f"{auth.base_url}/main_ops/download", {"bucket_name": bucket, "object_name": key}),
        # 3) alternativa: main_ops download kot path
        (f"{auth.base_url}/main_ops/download/{bucket}/{key}", None),
    ]

    last_err = None
    for url, params in candidates:
        try:
            r = requests.get(url, headers=headers, params=params)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                return
            last_err = f"{r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(
        f"Download failed for bucket='{bucket}' key='{key}'. Last error: {last_err}"
    )


def upload_path_as_object(auth: HumaineAuth, bucket: str, key: str, local_path: str) -> None:
    # 1) poskrbi, da imamo buckets list
    try:
        auth.list_buckets()
    except Exception:
        pass

    # 2) select bucket (to je nujno za main_ops/upload)
    auth.select_bucket(bucket)

    # 3) PRAVILEN upload endpoint: /main_ops/upload (prek auth.upload_object)
    # key = object_name
    auth.upload_object(key, local_path)
