-- Drop the table if it exists
DROP TABLE IF EXISTS `users`;

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS mentalstress;

-- Switch to the created database
USE mentalstress;

-- Create the user table
CREATE TABLE `users` (
    `name` VARCHAR(225),
    `email` VARCHAR(225),
    `password` VARCHAR(225),
    `phone number` VARCHAR(225),`age` VARCHAR(225)
    )