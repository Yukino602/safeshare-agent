# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3

from safeshare_agent.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys support
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Creates tables if they do not exist and seeds initial users."""
    conn = get_db_connection()
    try:
        with conn:
            # 1. Users table (3NF)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );
            """)

            # 2. Expenses table (3NF)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    amount REAL NOT NULL,
                    payer_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(payer_id) REFERENCES users(user_id)
                );
            """)

            # 3. Splits table (3NF - resolves composite relationship)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS splits (
                    expense_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    PRIMARY KEY(expense_id, user_id),
                    FOREIGN KEY(expense_id) REFERENCES expenses(expense_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)

            # Seed default users
            default_users = ["Tan", "Long Hei", "Alice", "Bob"]
            for name in default_users:
                conn.execute("INSERT OR IGNORE INTO users (name) VALUES (?);", (name,))
    finally:
        conn.close()


def get_known_users() -> dict[str, int]:
    """Retrieves all users as a mapping of name (lowercase) -> user_id."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name FROM users;")
        rows = cursor.fetchall()
        return {row["name"].lower(): row["user_id"] for row in rows}
    finally:
        conn.close()


def add_user_if_not_exists(name: str) -> int:
    """Inserts a new user if not exists and returns their user_id."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?);", (name,))
            cursor.execute("SELECT user_id FROM users WHERE name = ?;", (name,))
            row = cursor.fetchone()
            return row["user_id"]
    finally:
        conn.close()


def insert_expense(
    description: str,
    total_amount: float,
    payer_id: int,
    splits: list[tuple[int, float]],
) -> int:
    """Inserts an expense and its splits transactionally, returning the expense_id."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            # Insert the expense
            cursor.execute(
                "INSERT INTO expenses (description, amount, payer_id) VALUES (?, ?, ?);",
                (description, total_amount, payer_id),
            )
            expense_id = cursor.lastrowid

            # Insert splits
            for user_id, split_amount in splits:
                cursor.execute(
                    "INSERT INTO splits (expense_id, user_id, amount) VALUES (?, ?, ?);",
                    (expense_id, user_id, split_amount),
                )
            return expense_id
    finally:
        conn.close()
