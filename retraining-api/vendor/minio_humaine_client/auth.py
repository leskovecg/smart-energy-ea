import requests

class HumaineAuth:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.token = None
        self.buckets = []
        self.bucket_name = None

    def login(self, username, password):
        url = f"{self.base_url}/auth/auth"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(url, headers=headers, data={"username": username, "password": password})
        response.raise_for_status()
        self.token = response.json()["access_token"]
        return self.token

    def headers(self):
        if not self.token:
            raise ValueError("Not authenticated")
        return {"Authorization": f"Bearer {self.token}"}

    def list_buckets(self):
        url = f"{self.base_url}/main_ops/buckets"
        response = requests.get(url, headers=self.headers())
        response.raise_for_status()
        names = [item['name'] for item in response.json()["buckets"]]
        self.buckets = names
        return names

    def select_bucket(self, bucket_name):
        if bucket_name not in self.buckets:
            raise ValueError(f"Bucket '{bucket_name}' not found.")
        self.bucket_name = bucket_name
        return bucket_name

    def list_objects(self):
        if not self.bucket_name:
            raise ValueError("No bucket selected.")
        url = f"{self.base_url}/main_ops/objects/{self.bucket_name}"
        response = requests.get(url, headers=self.headers())
        response.raise_for_status()
        return response.json()["objects"]

    def upload_object(self, object_name, filepath):
        if not self.bucket_name:
            raise ValueError("No bucket selected.")
        url = f"{self.base_url}/main_ops/upload"
        files = {
            "bucket_name": (None, self.bucket_name),
            "object_name": (None, object_name),
            "file": open(filepath, "rb")
        }
        with open(filepath, "rb") as f:
            response = requests.post(url, headers=self.headers(), files=files)
        response.raise_for_status()
        return response.json()

    def update_metadata(self, object_name, metadata):
        if not self.bucket_name:
            raise ValueError("No bucket selected.")
        url = f"{self.base_url}/main_ops/update_metadata/{self.bucket_name}/{object_name}"
        response = requests.patch(url, headers=self.headers(), json=metadata)
        response.raise_for_status()
        return response.json()

    def get_metadata(self, object_name):
        if not self.bucket_name:
            raise ValueError("No bucket selected.")
        url = f"{self.base_url}/main_ops/metadata/{self.bucket_name}/{object_name}"
        response = requests.get(url, headers=self.headers())
        response.raise_for_status()
        return response.json()

