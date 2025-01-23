
-- CREATE TABLE IF NOT EXISTS users (
--   id INTEGER PRIMARY KEY AUTOINCREMENT,
--   abo_id TEXT UNIQUE NOT NULL,
--   email TEXT UNIQUE NOT NULL,
--   name TEXT NOT NULL,
--   phone TEXT,
--   password TEXT NOT NULL,
--   created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
--   chat_number INTEGER DEFAULT 0
-- );

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Authentication Details
  abo_id TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  
  -- Personal Details
  full_name TEXT NOT NULL,
  date_of_birth TEXT,
  gender TEXT,
  ethnicity TEXT,
  profile_picture BLOB,
  
  -- Communication Details
  mobile_number TEXT,
  email TEXT UNIQUE NOT NULL,
  address TEXT,
  
  -- Identity Details
  national_id TEXT,
  identity_proof BLOB,
  
  -- Sponsor Details
  sponsor_abo_id TEXT,
  sponsor_abo_name TEXT,
  
  -- Payment Details
  card_number TEXT,
  card_name TEXT,
  
  -- Final Verification
  id_proof_image BLOB,
  
  -- Metadata
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  chat_number INTEGER DEFAULT 0
);


CREATE TABLE IF NOT EXISTS chats (
  chat_id TEXT PRIMARY KEY,
  abo_id TEXT NOT NULL,
  snippet TEXT NOT NULL,
  last_updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (abo_id) REFERENCES users(abo_id)
);

CREATE TABLE IF NOT EXISTS messages (
  msg_id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL,
  user_content TEXT NOT NULL,
  bot_content TEXT NOT NULL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
);