import sqlite3
import pandas as pd

conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("📄 Tables:")
for name in tables:
    print("-", name[0])

# Print contents of each table (limit to 5 rows for readability)
for name in tables:
    table_name = name[0]
    print(f"\n🔍 {table_name} (top 5 rows):")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error reading {table_name}: {e}")

conn.close()
