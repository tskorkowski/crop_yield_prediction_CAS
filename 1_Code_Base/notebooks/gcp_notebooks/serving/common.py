from google.cloud import storage
import google.auth
from serving.constants import BUCKET

def data_init():
    """Authenticate and initialize Earth Engine with the default credentials."""
    credentials, project = google.auth.default()

def list_blobs_with_prefix(prefix, bucket=BUCKET):
    """Lists all the blobs in the bucket that begin with the prefix."""
    storage_client = storage.Client()
    return storage_client.list_blobs(bucket, prefix=prefix)