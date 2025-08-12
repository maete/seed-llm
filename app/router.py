import os
import re
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ValidationError, validator
from typing import List, Union, Optional

sys.path.append(os.path.dirname(__file__))
import llm_utils

logger = logging.getLogger(__name__)
router = APIRouter()

SEED_API = os.getenv("SEED_API", "http://localhost:80")

class LLMRequest(BaseModel):
    prompt: str
    llm_type: str
    property_ids: Optional[List[Union[int, str]]] = []
    columns: Optional[List[str]] = None  
    cycle_id: int
    data: Optional[list] = None

    @validator('property_ids', pre=True)
    def filter_none_values(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [item for item in v if item is not None]
        return v   

@router.post("/query_llm")
async def query_llm(request: Request):

    ############# Post Request Processing/Token Handling ###################
    body = await request.body()
    logger.info(f"Raw request body: {body.decode()}")

    try:
        json_data = json.loads(body.decode())
        logger.info(f"Parsed JSON: {json_data}")
        req = LLMRequest(**json_data)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw_token = request.headers.get("Authorization")
    if not raw_token:
        raise HTTPException(status_code=401, detail="Missing SEED token")

    if raw_token.startswith("Token eyJ"):
        seed_token = raw_token.replace("Token", "Bearer", 1)
    elif raw_token.startswith("Bearer "):
        seed_token = raw_token
    else:
        raise HTTPException(status_code=401, detail="Invalid token format")

    headers = {"Authorization": seed_token}

    # Get user/org info
    user_info_resp = requests.get(f"{SEED_API}/api/v3/users/current/", headers=headers, timeout=5)
    if user_info_resp.status_code != 200:
        raise HTTPException(status_code=user_info_resp.status_code, detail="Authentication failed with SEED")

    user_info = user_info_resp.json()
    user_id = user_info["id"]
    org_id = user_info.get("org_id")

    logger.debug("Fetching properties for organization")

    ############# Retrieving Data ###################

    all_property_data = []
    page = 1
    per_page = 100

    while True:
        properties_url = f"{SEED_API}/api/v3/properties/?organization_id={org_id}&per_page={per_page}&page={page}&cycle={req.cycle_id}"
        resp = requests.get(properties_url, headers=headers, timeout=10)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to retrieve properties")

        data = resp.json()
        page_results = data.get("results", [])
        if not page_results:
            break

        all_property_data.extend(page_results)

        # Break when we hit the last page
        if not data.get("pagination", {}).get("has_next", False):
            break

        page += 1

    ################ LLM Query #############################

    # Building Query
    df = pd.DataFrame(all_property_data)
    query = req.prompt
    categories = req.columns
    response_text = llm_utils.query_llm(df, query, categories)

     ############### LLM Query Cleaning Response ###########
    
    # Parse LLM response to dataframe
    rows = [line.split(',', 2) for line in response_text.strip().split('\n')]
    new_df = pd.DataFrame(rows, columns=["id", "LLM_Match", "LLM_Rationale"])
    new_df["id"] = new_df["id"].apply(lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else None)
    new_df = new_df.dropna(subset=["id"])

    # Merge with original data
    merged_df = df.merge(new_df, on="id", how="left")

    # Drop columns that are entirely empty: only contain NaN, [], or {}
    def is_entirely_empty(col):
        return all(pd.isna(x) or x == [] or x == {} for x in col)

    empty_cols = [col for col in merged_df.columns if is_entirely_empty(merged_df[col])]
    merged_df.drop(columns=empty_cols, inplace=True)

    # Drop columns where all values are 0
    def is_all_zero(col):
        return pd.api.types.is_numeric_dtype(col) and (col == 0).all()

    zero_cols = [col for col in merged_df.columns if is_all_zero(merged_df[col])]
    merged_df.drop(columns=zero_cols, inplace=True)

    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df = merged_df.fillna("Unknown")
    merged_df.columns = merged_df.columns.str.replace(r'_\d+$', '', regex=True)

    # Reorder columns to put LLM results after 'id'
    cols = merged_df.columns.tolist()
    for col in ["LLM_Match", "LLM_Rationale"]:
        if col in cols:
            cols.remove(col)
    id_index = cols.index("id") + 1
    for i, col in enumerate(["LLM_Match", "LLM_Rationale"]):
        cols.insert(id_index + i, col)
    merged_df = merged_df[cols]

    if "__text__" in merged_df.columns:
        merged_df.drop(columns=["__text__"], inplace=True)

    return {
        "user_id": user_id,
        "organization_id": org_id,
        "llm_type": req.llm_type,
        "response": merged_df.to_dict(orient="records"),
    }
