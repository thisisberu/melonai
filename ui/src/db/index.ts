// import Database from 'better-sqlite3';
// import { readFileSync } from 'fs';
// import { join } from 'path';

// import { fileURLToPath } from 'url';
// import { dirname } from 'path';

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = dirname(__filename);

// const db = new Database('chat.db');
// // Initialize database with schema
// const schema = readFileSync(join(__dirname, 'schema.sql'), 'utf-8');
// db.exec(schema);

// export default db;


import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Create database in a specific location relative to this file
const dbPath = join(__dirname, 'chat.db');
const db = new Database(dbPath);

// Initialize database with schema
try {
    const schema = readFileSync(join(__dirname, 'schema.sql'), 'utf-8');
    db.exec(schema);
} catch (error) {
    console.error('Error initializing database:', error);
}

export default db;
