#!/usr/bin/env python3
"""
Initialize metadata SQLite database for the time-series transformer project.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_metadata_database(db_path: str = "data/metadata/metadata.db"):
    """
    Create and initialize the metadata SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
    """
    # Ensure directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database (creates if doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create data_sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                base_url TEXT,
                api_key_required BOOLEAN DEFAULT FALSE,
                rate_limit_per_second INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tickers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create data_downloads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_id INTEGER NOT NULL,
                data_source_id INTEGER NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                records_count INTEGER,
                file_path TEXT,
                file_size_bytes INTEGER,
                download_duration_seconds REAL,
                status TEXT CHECK(status IN ('pending', 'completed', 'failed', 'partial')) DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker_id) REFERENCES tickers (id),
                FOREIGN KEY (data_source_id) REFERENCES data_sources (id)
            )
        """)
        
        # Create feature_engineering_jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_engineering_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_id INTEGER NOT NULL,
                input_file_path TEXT NOT NULL,
                output_file_path TEXT,
                features_config TEXT, -- JSON string of feature configuration
                processing_duration_seconds REAL,
                status TEXT CHECK(status IN ('pending', 'completed', 'failed')) DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker_id) REFERENCES tickers (id)
            )
        """)
        
        # Create data_quality_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_id INTEGER NOT NULL,
                data_source_id INTEGER NOT NULL,
                metric_date DATE NOT NULL,
                total_records INTEGER,
                missing_records INTEGER,
                outlier_records INTEGER,
                duplicate_records INTEGER,
                quality_score REAL, -- 0.0 to 1.0
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker_id) REFERENCES tickers (id),
                FOREIGN KEY (data_source_id) REFERENCES data_sources (id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickers_symbol ON tickers(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_downloads_ticker_date ON data_downloads(ticker_id, start_date, end_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_downloads_status ON data_downloads(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_ticker_date ON data_quality_metrics(ticker_id, metric_date)")
        
        # Insert default data sources
        default_sources = [
            ('yahoo_finance', 'Yahoo Finance API', 'https://query1.finance.yahoo.com', False, 5),
            ('alpha_vantage', 'Alpha Vantage API', 'https://www.alphavantage.co', True, 1),
            ('news_api', 'News API for sentiment data', 'https://newsapi.org', True, 1)
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO data_sources (name, description, base_url, api_key_required, rate_limit_per_second)
            VALUES (?, ?, ?, ?, ?)
        """, default_sources)
        
        # Insert sample tickers (S&P 500 major components)
        sample_tickers = [
            ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
            ('MSFT', 'Microsoft Corporation', 'Technology', 'Software'),
            ('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Content & Information'),
            ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'Internet & Direct Marketing Retail'),
            ('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Automobiles'),
            ('META', 'Meta Platforms Inc.', 'Technology', 'Interactive Media & Services'),
            ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors'),
            ('JPM', 'JPMorgan Chase & Co.', 'Financials', 'Banks'),
            ('JNJ', 'Johnson & Johnson', 'Health Care', 'Pharmaceuticals'),
            ('V', 'Visa Inc.', 'Financials', 'Data Processing & Outsourced Services')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO tickers (symbol, name, sector, industry)
            VALUES (?, ?, ?, ?)
        """, sample_tickers)
        
        # Commit changes
        conn.commit()
        logger.info(f"Successfully initialized metadata database at {db_path}")
        
        # Print summary
        cursor.execute("SELECT COUNT(*) FROM data_sources")
        sources_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickers")
        tickers_count = cursor.fetchone()[0]
        
        logger.info(f"Database contains {sources_count} data sources and {tickers_count} tickers")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_metadata_database()