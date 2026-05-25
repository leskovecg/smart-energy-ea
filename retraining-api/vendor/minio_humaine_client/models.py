import requests

def upload_model(auth, filepath):
    with open(filepath, "rb") as f:
        response = requests.post(
            f"{auth.base_url}/data/upload",
            headers=auth.headers(),
            files={"file": f}
        )
    response.raise_for_status()
    return response.json()

def download_model(auth, object_key, dest_path):
    response = requests.get(
        f"{auth.base_url}/data/download",
        headers=auth.headers(),
        params={"key": object_key}
    )
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
