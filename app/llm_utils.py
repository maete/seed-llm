import numpy as np
import pandas as pd
import difflib
import requests
import json
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


# =========================
# Base Category Definitions
# =========================

BASE_COLUMN_ALIASES = {
    # ID Columns
    "id": ["id", "ABID", "PM Property ID", "Custom id 1"],

    # Name / Title
    "name": ["Property Name", "Name", "Web Building Name"],

    # Address-related
    "address": [
        "Address", "Address 1", "Address 2", "Address Line 1", "City", "State", 
        "State/Province", "Postal Code", "Zipcode", "PM City", "Web Address", "County"
    ],

    # Building Type / Use Type
    "property_type and ameneties": [
        "Primary Property Type - Portfolio Manager-Calculated", 
        "Primary Property Type - Self Selected",
        "Largest Property Use Type", 
        "2nd Largest Property Use Type",
        "3rd Largest Property Use Type",
        "List of All Property Use Types (GFA) (ft2)",
        "Web Property Type"
    ],

    # Floor Area
    "floor_area": [
        "Property GFA - Calculated (Buildings and Parking) (ft2)",
        "Property GFA - Calculated (Buildings) (ft2)",
        "Property GFA - Calculated (Parking) (ft2)",
        "Property GFA - Self-Reported (ft2)",
        "Largest Property Use Type - Gross Floor Area (ft2)",
        "2nd Largest Property Use Type - Gross Floor Area (ft2)",
        "3rd Largest Property Use Type - Gross Floor Area (ft2)",
        "Irrigated Area (ft2)",
        "Web Bldg SF"
    ],

    # Year Info
    "year_built": ["Year Built"],
    "year_reporting": ["Year Ending", "Audit Year"],

    # ENERGY STAR / EUI / Energy Performance
    "energy_star_score": [
        "ENERGY STAR Score", "National Median ENERGY STAR Score"
    ],
    "EUI": [
        "Site EUI (kBtu/ft2)", 
        "Source EUI (kBtu/ft2)",
        "Weather Normalized Site EUI (kBtu/ft2)",
        "Weather Normalized Source EUI (kBtu/ft2)",
        "National Median Site EUI (kBtu/ft2)"
    ],
    "site_energy_use": [
        "Site Energy Use (kBtu)", 
        "Site Energy Use - Adjusted to Current Year (kBtu)", 
        "Weather Normalized Site Energy Use (kBtu)"
    ],
    "source_energy_use": [
        "Source Energy Use (kBtu)",
        "Source Energy Use - Adjusted to Current Year (kBtu)",
        "Weather Normalized Source Energy Use (kBtu)"
    ],

    # GHG Emissions
    "ghg_emissions": [
        "Total (Location-Based) GHG Emissions (Metric Tons CO2e)",
        "National Median Total (Location-Based) GHG Emissions (Metric Tons CO2e)",
        "Total (Location-Based) GHG Emissions Intensity (kgCO2e/ft2)"
    ],

    # Water
    "water_use": [
        "Water Use (All Water Sources) (kgal)",
        "Indoor Water Use (All Water Sources) (kgal)",
        "Outdoor Water Use (All Water Sources) (kgal)"
    ],
    "water_eui": [
        "Water Use Intensity (All Water Sources) (gal/ft2)",
        "Indoor Water Use Intensity (All Water Sources) (gal/ft2)"
    ],

    # Fuel / Energy Types (sub metering)
    "natural_gas": ["Natural Gas Use (kBtu)", "Natural Gas Use (therms)"],
    "district_steam": ["District Steam Use (kBtu)"],
    "district_chilled_water": ["District Chilled Water Use (kBtu)"],
    "onsite_renewables": [
        "Electricity Use - Generated from Onsite Renewable Systems (kWh)",
        "Electricity Use - Generated from Onsite Renewable Systems and Exported (kWh)",
        "Electricity Use - Generated from Onsite Renewable Systems and Used Onsite (kBtu)",
        "Electricity Use - Generated from Onsite Renewable Systems and Used Onsite (kWh)",
        "Percent of Total Electricity Generated from Onsite Renewable Systems"
    ],
    "electricity_grid": [
        "Electricity Use - Grid Purchase (kWh)",
        "Electricity Use - Grid Purchase (kBtu)"
    ],
    "green_power": [
        "Green Power - Offsite (kWh)", 
        "Green Power - Onsite (kWh)", 
        "Green Power - Onsite and Offsite (kWh)"
    ],
    "fuel_oil": ["Fuel Oil #2 Use (kBtu)"],

    # Occupancy / Units
    "occupancy": ["Occupancy"],
    "unit_count": ["Multifamily Housing - Total Number of Residential Living Units"],

    # Labels / Metadata / Data Quality
    "data_quality": [
        "Data Quality Checker - Date Run", 
        "Data Quality Checker Run?", 
        "Default Values",
        "Estimated Values - Energy", 
        "Estimated Values - Water"
    ],
    "alerts": [
        "Alert - Energy - No meters selected for metrics",
        "Alert - Energy Meter has gaps",
        "Alert - Energy Meter has less than 12 full calendar months of data",
        "Alert - Energy Meter has overlaps",
        "Alert - Energy Meter has single entry more than 65 days",
        "Alert - Gross Floor Area is 0 ft2",
        "Alert - Property has no uses",
        "Alert - Water - No meters selected for metrics",
        "Alert - Water Meter has gaps",
        "Alert - Water Meter has less than 12 full calendar months of data",
        "Alert - Water Meter has overlaps"
    ],

    # Reporting / Submission
    "reporting_dates": ["Report Generation Date", "Report Submission Date"],
    "submitted_on_behalf": [
        "Report Submitted On Behalf Of - Email",
        "Report Submitted On Behalf Of - Name",
        "Report Submitted On Behalf Of - Organization",
        "Report Submitted On Behalf Of - Phone"
    ],
    "service_provider": ["Service and Product Provider"],
    "shared_by": ["Shared By Contact"],

    # Certification / Awards
    "certification": [
        "ENERGY STAR Certification - Year(s) Certified",
        "Third Party Certification",
        "Eligible for Certification for Report PED (Y/N)"
    ],

    # Administrative / Last Modified
    "admin_modified": [
        "Last Modified By - Electric Meters", "Last Modified Date - Electric Meters",
        "Last Modified By - Gas Meters", "Last Modified Date - Gas Meters",
        "Last Modified By - Non-Electric Non-Gas Energy Meters", "Last Modified Date - Non-Electric Non-Gas Energy Meters",
        "Last Modified By - Property", "Last Modified Date - Property",
        "Last Modified By - Waste Meters", "Last Modified Date - Waste Meters",
        "Last Modified By - Water Meters", "Last Modified Date - Water Meters"
    ],

    # Carbon Offsets / RECs
    "avoided_emissions": [
        "Avoided Emissions - Offsite Green Power (Metric Tons CO2e)",
        "Avoided Emissions - Onsite Green Power (Metric Tons CO2e)",
        "Avoided Emissions - Onsite and Offsite Green Power (Metric Tons CO2e)"
    ],
    "recs_percent": ["Percent of RECs Retained"],

    # Weather
    "weather": [
        "Cooling Degree Days (CDD) (degF)",
        "Heating Degree Days (HDD) (degF)"
    ],

    "notes": [
        "Property Notes", 
        "Property Notes2",
        "property_notes",
        "property_type",
    ],

    # Ignore Columns (non-useful for modeling)
    "ignore": [
        "Created", "Updated","Property Labels",
        "Energy Baseline Date", "Energy Current Date", "Water Baseline Date", "Water Current Date"
    ]
}

# =========================
# Category-Based Selection Rules
# =========================

CATEGORY_SELECTION_RULES = {
    "pool": ["water_use", "property_type and ameneties"],
    "solar": ["onsite_renewables", "green_power", "electricity_grid"],
    "energy efficiency": ["energy_star_score", "EUI", "site_energy_use"],
    "GHG emissions": ["ghg_emissions", "source_energy_use"],
    "year built": ["year_built"]
}

# =========================
# Column Mapping
# =========================

def auto_map_columns(df_columns, base_aliases, cutoff=0.6):

    assigned = {}

    for logical_name, aliases in base_aliases.items():
        matched_cols = []
        for col in df_columns:
            col_lower = col.lower()
            for alias in aliases:
                ratio = difflib.SequenceMatcher(None, col_lower, alias.lower()).ratio()
                if ratio >= cutoff or alias.lower() in col_lower:
                    matched_cols.append(col)
                    break
        if matched_cols:
            assigned[logical_name] = matched_cols
    return assigned

def get_merged_value(row, columns):
    values = []
    for col in columns:
        if col in row and pd.notnull(row[col]):
            values.append(str(row[col]))
    return ", ".join(values) if values else "Unknown"

def row_to_text_with_catch_all(row, column_mapping):
    parts = []
    mapped_cols_flat = [col for cols in column_mapping.values() for col in cols]
    catch_all_parts = []

    for logical_name, cols in column_mapping.items():
        merged_val = get_merged_value(row, cols)
        field_name = logical_name.replace('_', ' ').title()
        parts.append(f"{field_name}: {merged_val}")

    for col in row.index:
        val = row[col]
        is_array = isinstance(val, (list, np.ndarray))
        is_empty_array = is_array and len(val) == 0
        try:
            if (
                col not in mapped_cols_flat
                and not is_empty_array
                and not (is_array and pd.isnull(val).all())  # ensure multi-element nulls don’t crash
                and not (hasattr(val, "__len__") and len(val) == 0)  # protect against ambiguous sequences
                and pd.notnull(val) if not is_array else True  # allow arrays to pass
            ):
                catch_all_parts.append(f"{col}: {val}")
        except Exception as e:
            print(f"Skipping column {col} due to error: {e}")

    if catch_all_parts:
        parts.append("\nAdditional Details:\n" + "\n".join(catch_all_parts))

    return "\n".join(parts)

# =========================
# Category Selector Functions
# =========================

def fuzzy_category_selector(question_text):
    selected = []
    for keyword, categories in CATEGORY_SELECTION_RULES.items():
        if keyword in question_text.lower():
            selected.extend(categories)
    return selected

def llm_category_selector(question_text, logical_categories, ollama_url):
    prompt = f"""
You are analyzing building data.

Here are the available data categories:

{logical_categories}

Given this user question:

"{question_text}"

Select 1-2 most relevant categories that would help answer the question.
Return just a Python list of category names from the list above:

['column1', 'column2']

Don't return anything else. No intro, no explanation, ONLY the relevant columns.

"""
    try:
        response = requests.post(
            ollama_url,
            json={"model": "llama3", "prompt": prompt, "stream": False},
        )
        output = response.json()["response"]
        selected = eval(output.strip())
        return selected if isinstance(selected, list) else []
    except Exception as e:
        print(f"LLM category selector failed: {e}")
        return []

# =========================
# Full Pipeline
# =========================

def batch_rag_pipeline_by_category(
    query_text, 
    embedding_model, 
    vectorstore, 
    docs, 
    df, 
    column_mapping, 
    selected_categories,
    ollama_url="http://localhost:11434/api/generate", 
    batch_size=2,
    ):

    logical_categories = list(BASE_COLUMN_ALIASES.keys())

    if not selected_categories:
        selected_categories = fuzzy_category_selector(query_text)
        if not selected_categories:
            print("No match from fuzzy rules, falling back to LLM category selector...")
            selected_categories = llm_category_selector(query_text, logical_categories, ollama_url)

        if not selected_categories:
            print("Warning: No categories selected. Defaulting to all columns.")
            selected_categories = logical_categories

    selected_cols = []
    for cat in selected_categories:
        selected_cols.extend(column_mapping.get(cat, []))

    print(f"\n Using categories: {selected_categories}")
    print(f" Resolved columns: {selected_cols}\n")

    all_results = []
    total_docs = len(docs)

    for batch_start in tqdm(range(0, total_docs, batch_size)):

        batch_docs = docs[batch_start:batch_start + batch_size]
        context_chunks = []

        for doc in batch_docs:
            building_id = doc.metadata.get("building_id")
            row_data = df.loc[df["id"] == building_id]
            if not row_data.empty:
                row = row_data.iloc[0]
                filtered_data = "\n".join([f"{col}: {row.get(col, '')}" for col in selected_cols])
                context_chunks.append(f"Building ID {building_id}:\n{filtered_data}")

        retrieved_context = "\n\n".join(context_chunks)
        if not retrieved_context.strip():
            continue

        prompt = f"""
You are a strict data extraction bot. Do not provide any explanations, summaries, or introductory text. Your ONLY job is to extract the requested structured data.

The data below contains building property records.

---

Building Records:
{retrieved_context}

---

Question:
{query_text}

Output Format:

Return buildings, one per line, in CSV format with this exact structure (no header row):

building_id,Yes/No Boolean,reason

Where:
- building_id = The numeric ID of the building
- Yes/No Boolean = A Yes if it does match critiera, no if it does not
- reason = A short reason (one sentence max) why the yes/no selection was made

Do not return any non-matching buildings.
Do not return blank lines.
Do not include any JSON, markdown, or explanation text.
Do not include a header row.
Do not include any brackets or code formatting.

Example Output:

123456,Yes,records mention having x
789012,No,no mention in records

Now generate the output:
"""

        response = requests.post(
            ollama_url,
            json={"model": "llama3", "prompt": prompt, "stream": True},
        )

        batch_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                batch_response += data.get("response", "")
        all_results.append(batch_response)

    final_result = "\n".join(all_results)
    return final_result


def query_llm(df, query, selected_categories = None):

    # Step 1: Column mapping
    column_mapping = auto_map_columns(df.columns, BASE_COLUMN_ALIASES)

    # Step 2: Generate __text__ for embedding
    df["__text__"] = df.apply(lambda row: row_to_text_with_catch_all(row, column_mapping), axis=1)

    # Step 3: Embed documents
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    docs = [Document(page_content=row["__text__"], metadata={"building_id": row.get("id", "Unknown")}) for _, row in tqdm(df.iterrows(), total=len(df))]

    # Step 4: FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Step 5: Run batch RAG
    results = batch_rag_pipeline_by_category(query, embedding_model, vectorstore, docs, df, column_mapping, selected_categories)
    
    return results