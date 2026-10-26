"""
Database for Role-Based Access Control (RBAC) for sEMG Data

Security mechanisms implemented as sEMG data can be considered medical data

Researcher - read, write, and delete from all tables
Clinician - read to all tables
Patient - can only view their own data  - This program assumes that each patient has individual databases because we only collected data from one individual.

#SQLite extension must be installed - this extension does does not have built-in support for role-based access controls, so they are implemented at the application level.

Requires the following libraries: pandas, sqlite3
"""

import pandas as pd
import sqlite3

#Connect to the database file, or create it if it doesn't exist
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

#List of CSV files and their target table names
csv_files = {
    "table_one": "csv_data/01.csv",
    "table_two": "csv_data/02.csv",
    "table_three": "csv_data/03.csv",
    "table_four": "csv_data/04.csv",
    "table_five": "csv_data/05.csv",
    "table_six": "csv_data/06.csv",
    "table_seven": "csv_data/07.csv"
}

#Write each CSV to the database, replacing the table if it already exists
for table_name, csv_path in csv_files.items():
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

#create the necessary tables for RBAC

cursor.execute("""
    -- prevents an error from being thrown if the table already exists
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL
    );
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_roles (
    --many-to-many relationship between users and roles
    user_id INTEGER,
    role_id INTEGER,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS role_permissions (
    --many-to-many relationship between roles and permissions
    role_id INTEGER,
    permission_id INTEGER,
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);
""")

#Creates permissions
cursor.executemany("INSERT OR IGNORE INTO permissions (name) VALUES (?);", [('edit_data',), ('view_data',), ('delete_data',)])
    
#Creates roles
cursor.execute("INSERT OR IGNORE INTO roles (name) VALUES ('Patient');")
patient_role_id = cursor.lastrowid # assign the role ID to the last inserted row ID
cursor.execute("INSERT OR IGNORE INTO roles (name) VALUES ('Researcher');")
researcher_role_id = cursor.lastrowid 
cursor.execute("INSERT OR IGNORE INTO roles (name) VALUES ('Clinician');")
clinician_role_id = cursor.lastrowid 
    
#Gives permissions unique IDs
cursor.execute("SELECT id FROM permissions WHERE name IN ('view_data');")
permission_ids_view = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT id FROM permissions WHERE name IN ('edit_data');")
permission_ids_edit = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT id FROM permissions WHERE name IN ('delete_data');")
permission_ids_delete = [row[0] for row in cursor.fetchall()]
    
#Links permissions to roles
patient_role_permissions = [(patient_role_id, perm_id) for perm_id in permission_ids_view]
cursor.executemany("INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?);", patient_role_permissions)

researcher_role_permissions = [(researcher_role_id, perm_id) for perm_id in permission_ids_view + permission_ids_edit + permission_ids_delete]
cursor.executemany("INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?);", researcher_role_permissions)

clinician_role_permissions = [(clinician_role_id, perm_id) for perm_id in permission_ids_view + permission_ids_edit]
cursor.executemany("INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?);", clinician_role_permissions)

#Creates users and assigns them to roles
cursor.execute("INSERT OR IGNORE INTO users (username) VALUES ('alice_patient');")
patient_user_id = cursor.lastrowid
cursor.execute("INSERT OR IGNORE INTO users (username) VALUES ('bob_researcher');")
researcher_user_id = cursor.lastrowid
cursor.execute("INSERT OR IGNORE INTO users (username) VALUES ('charlie_clinician');")
clinician_user_id = cursor.lastrowid
    
cursor.execute("INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?);", (patient_user_id, patient_role_id))
cursor.execute("INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?);", (researcher_user_id, researcher_role_id))
cursor.execute("INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?);", (clinician_user_id, clinician_role_id))
    
conn.commit()


def user_has_permission(username, required_permission):
    cursor = conn.cursor()
    
    # SQL query checking the chain: User -> User_Roles -> Role_Permissions -> Permissions
    query = """
    SELECT 1 FROM users u
    JOIN user_roles ur ON u.id = ur.user_id
    JOIN role_permissions rp ON ur.role_id = rp.role_id
    JOIN permissions p ON rp.permission_id = p.id
    WHERE u.username = ? AND p.name = ?
    LIMIT 1;
    """
    
    cursor.execute(query, (username, required_permission))
    return cursor.fetchone() is not None

# Test actions - for validation purposes only
has_view = user_has_permission('alice_patient', 'view_data')
print(f"Can Alice view? {has_view}")  # Output: True
has_edit = user_has_permission('alice_patient', 'edit_data')
print(f"Can Alice edit? {has_edit}")  # Output: False
has_delete = user_has_permission('alice_patient', 'delete_data')
print(f"Can Alice delete? {has_delete}")  # Output: False

has_view = user_has_permission('bob_researcher', 'view_data')
print(f"Can Bob view? {has_view}")  # Output: True
has_edit = user_has_permission('bob_researcher', 'edit_data')
print(f"Can Bob edit? {has_edit}")  # Output: True
has_delete = user_has_permission('bob_researcher', 'delete_data')
print(f"Can Bob delete? {has_delete}")  # Output: True

has_view = user_has_permission('charlie_clinician', 'view_data')
print(f"Can Charlie view? {has_view}")  # Output: True
has_edit = user_has_permission('charlie_clinician', 'edit_data')
print(f"Can Charlie edit? {has_edit}")  # Output: True
has_delete = user_has_permission('charlie_clinician', 'delete_data')
print(f"Can Charlie delete? {has_delete}")  # Output: False

conn.close()