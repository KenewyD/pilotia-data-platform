import os
from sqlalchemy import create_engine
from src.data_generator import ensure_data

def main():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "pilotia")
    user = os.getenv("POSTGRES_USER", "pilotia")
    password = os.getenv("POSTGRES_PASSWORD", "pilotia")
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
    df = ensure_data()
    df.to_sql("fact_activity", engine, if_exists="replace", index=False, chunksize=5000, method="multi")
    print(f"{len(df):,} lignes chargées dans fact_activity.")

if __name__ == "__main__":
    main()
