from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_id: str
    firebase_app_name: str = "fluffy-cafe-loyalty"
    firebase_cert_path: str = "/secrets/firebase-service-account.json"
    firestore_users_collection: str = "users"
    firestore_receipts_collection: str = "receipts"
    storage_receipts_bucket: str
    documentai_processor_id: str | None = None
    documentai_location: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
