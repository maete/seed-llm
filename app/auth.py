import os
from fastapi import HTTPException

VALID_TOKEN = os.getenv("ACCESS_TOKEN")

def verify_token(token: str):
    if token != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token