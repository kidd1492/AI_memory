import _sqlite3

database_path = "rag_store.db"
conn = _sqlite3.connect(database_path)

# If you want to work on a specific table or perform operations that require opening an existing connection,
# you would do something like this:
cursor = conn.cursor()
cursor.execute("SELECT content FROM messages") 
rows = cursor.fetchall() 
for row in rows: print(f"\n{row}")
'''
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    for column in schema:
        print(column)
'''
conn.close()