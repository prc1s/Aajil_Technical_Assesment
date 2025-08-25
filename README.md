# Purchase Order Item Categorization

⚠️ This `README.md` was rewritten by ChatGPT from my original notes (`README_raw.md`)  
for clarity and professionalism. Both are included in this repo for transparency.  

- **README_raw.md** → my original notes, messy exploration, and thinking process.  
- **README.md** (this file) → polished rewrite for clarity, structure, and readability.

---

## Running the pipeline

1.  Activate your virtual environment  
   ```bash
   source venv/bin/activate    # Mac/Linux
   venv\Scripts\activate       # Windows

2.  Install dependencies

    pip install -r requirements.txt

3.  Run all stages

    python main.py

4.  Start MLflow UI to inspect runs

    mlflow ui

Outputs are saved in the artifacts/ directory

---

## Goal

Develop an end-to-end pipeline to categorize messy multilingual purchase order data,  
balancing **speed, professionalism, and scalability** with modular code.

---

## Project Structure

experimentation_notebooks/ # notebooks for trials and EDA
src/datascience/
│── init.py # logger setup
│── utils/common.py # helper functions (YAML, ensure_dir, etc.)
│── components/ # ingestion, preprocessing, categorization, analysis
│── pipeline/ # orchestrating pipeline stages
config/config.yaml # configurations for all stages


---

## Workflow

### 1. Data Ingestion
- Converted `purchase-order-items.xlsx` → CSV.
- Added constants, configs, and `01_data_ingestion` notebook.
- Config manager retrieves `data_ingestion_config` and runs ingestion.

### 2. Exploratory Data Analysis
Key findings:
- `Project ID` useless (all null).  
- `Item Name` & `Product ID` have 240 nulls (~7.6%).  
- `Tax ID` has 65 nulls (~2%).  
- IDs had no business value.  
- 3,085 SAR rows, 65 USD rows. USD spend is small but included.  
- Added **Unit Price** (`Total Bcy / Quantity`) for sanity checks.  
- Rows with `Quantity>0` & `Total Bcy=0` dropped.  
- `Total Bcy` ≈ `Sub Total Bcy` (redundant → dropped).  
- Null Item Names renamed to **Unknown Product** (≈9.9% revenue).  
- Patterns:
  - Most revenue: steel, concrete, rubber.  
  - Most expensive (unit price): capital items/services.  
  - Highest quantities: cheap construction materials.

### 3. Preprocessing
- Dropped unhelpful columns (`Purchase Order ID`, `Product ID`, `Account ID`, `Tax ID`, `Project ID`, `Item ID`, `Sub Total Bcy`).  
- Converted USD → SAR.  
- Filled missing names with “Unknown Product.”  
- Dropped invalid rows (`Quantity>0` & `Total Bcy=0`).  
- Added `Unit Price`.  
- Normalization:
  - **EN:** lowercase, trim, remove punctuation (keep specs like `200x200`).  
  - **AR:** remove tashkeel, tatweel, unify letters.

### 4. Categorization
- Tried:
  - **Keyword rules** → precise, but poor coverage.  
  - **LLM dictionary expansion** → too costly and slow.  
  - **Multilingual Hugging Face model** → better coverage.  
- Logged experiments in MLflow.  
- Hybrid strategy chosen:
  - Rules for high-spend items.  
  - Embeddings + clustering for long tail.  
- Issues:
  - “Other” bucket too large.  
  - Rebar/Pipes clusters too noisy.

### 5. Insights
- **Spend share**:
  - Rebar & Steel ≈ 31% (~48.8M SAR).  
  - Electrical & Cables ≈ 17% (~26.9M SAR).  
  - Other ≈ 13.9% (~21.8M SAR).  
- **Consistency**:
  - PPE & Safety, Fasteners: low variation → reliable.  
  - Rebar & Steel, Pipes & Fittings: high variation → need sub-categories.  
- **Confidence vs consistency**:
  - High confidence + high variation = misleading clusters.  
  - Shows where rules/translation are needed.

---

## With More Time
- Sub-categorize engineering materials (Rebar by diameter/length; Pipes by material/class).  
- Hybrid rule + clustering system with regex for measurements.  
- Cross-language unification (merge AR/EN duplicates).  
- Human-in-the-loop review of low-confidence clusters.  
- Stronger data validation (schema, integrity checks).  
- More parameter tuning & alternative clustering techniques.
