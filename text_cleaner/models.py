import uuid
from pydantic import BaseModel


class UploadCV(BaseModel):
    id: uuid.uuid4
    name: list[str]
    address: list[str]
    phone: list[str]
    email: list[str]
    website: list[str]
