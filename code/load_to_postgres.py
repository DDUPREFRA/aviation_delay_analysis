# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

import pandas as pd
import psycopg2
from pathlib import Path
from sqlalchemy import create_engine, text


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main cleaned file loaded into PostgreSQL.
PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Database connection settings.
# Keep credentials outside the source code so they are not published to Git.
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "flight_delays")
TABLE_NAME = "flights"


def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 3. LOAD CLEANED CSV
# ============================================================

print_section("LOAD CLEANED DATA TO POSTGRESQL")
print(f"Input file: {PATH}")

# Read only a small sample into pandas.
# The full file has tens of millions of rows, so pandas should not load it all.
df_sample = pd.read_csv(PATH, nrows=1000, low_memory=False)
input_columns = len(df_sample.columns)


# ============================================================
# 4. CONNECT TO POSTGRESQL
# ============================================================

# Create the PostgreSQL connection.
# SQLAlchemy is the library pandas uses to send the dataframe to the database.
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# ============================================================
# 5. UPLOAD DATA
# ============================================================

print()
print(f"Uploading to PostgreSQL table: {TABLE_NAME}")

# Create an empty table with the CSV column names.
# if_exists="replace" drops the old table and recreates it.
print("Creating empty table...")
df_sample.head(0).to_sql(
    TABLE_NAME,
    engine,
    if_exists="replace",
    index=False
)

print("Empty table created.")
print("Starting fast COPY upload...")

# Open a direct connection to PostgreSQL.
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Create a cursor, which is the object that sends SQL commands to PostgreSQL.
cur = conn.cursor()

# Open CSV File.
with open(PATH, "r") as f:

    # Copy data into the flights table
    # Read the data from this CSV file
    # First row is a header row
    cur.copy_expert(
        f"""
        COPY {TABLE_NAME}
        FROM STDIN
        WITH CSV HEADER
        """,
        f
    )

# Save the upload permanently
conn.commit()

# Close the PostgreSQL cursor and connection.
cur.close()
conn.close()

print("Upload finished!")


# ============================================================
# 6. VERIFY ROW COUNT
# ============================================================

# Check how many rows are in the PostgreSQL table after the upload.
with engine.connect() as conn:
    result = conn.execute(text(f'SELECT COUNT(*) FROM "{TABLE_NAME}"'))
    count = result.scalar()

# Print the final upload summary.
print_section("POSTGRESQL LOAD SUMMARY")

print(f"Database:       {DB_NAME}")
print(f"Table:          {TABLE_NAME}")
print(f"Columns loaded: {input_columns}")

# This value comes from PostgreSQL after upload.
print(f"Verified rows:  {count:,}")
