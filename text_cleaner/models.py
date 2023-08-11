import uuid
from pydantic import BaseModel


class UploadCV(BaseModel):
    id: str
    name: list[str]
    address: list[str]
    phone: list[str]
    email: list[str]
    url: list[str]
    other: list[str]
