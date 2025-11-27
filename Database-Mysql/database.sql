-- ------------------------------------------------------------
-- Salescope minimal schema
-- Compatible with your current Flask app (signup/login + forgot password)
-- ------------------------------------------------------------

-- 1) create database
CREATE DATABASE IF NOT EXISTS salescope
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE salescope;

--  2) users table
-- -- Your code does:
-- INSERT INTO users (email, company, mobile, owner, password) VALUES (...)
-- SELECT * FROM users WHERE mobile = %
-- session["owner"] = user["owner"]
-- so we MUST keep these column names.
-- DROP TABLE IF EXISTS users;
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email   VARCHAR(255) NOT NULL,
  company VARCHAR(255) DEFAULT NULL,
  mobile  VARCHAR(15)  NOT NULL,
  owner   VARCHAR(255) DEFAULT NULL,
  password VARCHAR(512) NOT NULL,  -- werkzeug password hash goes here
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NULL ON UPDATE CURRENT_TIMESTAMP,

  -- your app should not allow two accounts with same email or mobile
  UNIQUE KEY uq_users_email (email),
  UNIQUE KEY uq_users_mobile (mobile),
  INDEX idx_users_mobile (mobile)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3) password_resets table
-- Your forgot-password code (the one I gave) uses:
--   INSERT INTO password_resets (user_id, token, expires_at) ...
--   SELECT * FROM password_resets WHERE token=%s AND used=0
--   UPDATE password_resets SET used=1 WHERE token=%s
DROP TABLE IF EXISTS password_resets;
CREATE TABLE password_resets (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  token VARCHAR(255) NOT NULL UNIQUE,
  expires_at DATETIME NOT NULL,
  used TINYINT(1) NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_pr_user (user_id),
  INDEX idx_pr_expires (expires_at),
  CONSTRAINT fk_pr_user FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- OPTIONAL: create a demo user manually (uncomment & edit)
-- NOTE: password must be a HASH from your Flask app
-- Just sign up from /signup and you don't need this.
-- ------------------------------------------------------------
-- INSERT INTO users (email, company, mobile, owner, password)
-- VALUES ('demo@example.com', 'Demo Company', '9999999999', 'Demo Owner',
--         'PASTE_HASH_FROM_FLASK_HERE');

-- ------------------------------------------------------------
-- OPTIONAL: cron / event to clean old tokens
-- DELETE FROM password_resets
-- WHERE used = 1 OR expires_at < UTC_TIMESTAMP();

SELECT COUNT(*) AS total_users FROM users;
SELECT id, email, company, mobile, owner, created_at FROM users;


