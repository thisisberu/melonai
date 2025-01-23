import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
// import { User } from '../models/User';

// import db from '../db';
import db from '../db/index.js'
import type { LoginCredentials, RegisterCredentials } from '../types/auth';

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// router.post('/register', async (req, res) => {
//   try {
//     console.log("inside register api",req.body); 
//     const { aboId, password,name,email, phone_number }: RegisterCredentials = req.body;

//     const existingUser = db.prepare('SELECT * FROM users WHERE abo_id = ?').get(aboId);
//     if (existingUser) {
//       return res.status(400).json({ error: 'ABO ID already exists' });
//     }

//     const hashedPassword = await bcrypt.hash(password, 10);
//     // const result = db.prepare('INSERT INTO users (abo_id, password, chat_number,name,email,phone) VALUES (?,?,?,?,?,?)').run(aboId, hashedPassword, 0);
//     // ... existing code ...
//     const result = db.prepare('INSERT INTO users (abo_id, password, chat_number, name, email, phone) VALUES (?, ?, ?, ?, ?, ?)').run(aboId, hashedPassword, 0, name, email, phone_number);
// // ... existing code ... 
//     const token = jwt.sign({ userId: result.lastInsertRowid }, JWT_SECRET, { expiresIn: '24h' });

//     // Get the full user data after insertion
//     const newUser = db.prepare('SELECT * FROM users WHERE id = ?').get(result.lastInsertRowid);

//     res.json({
//       user: {
//         id: newUser.id,
//         aboId: newUser.abo_id,
//         createdAt: new Date(newUser.created_at),
//         chat_number: newUser.chat_number,  // Added this line,
//         name: newUser.name, 
//         email: newUser.email,
//         phone: newUser.phone
//       },
//       token
//     });
//   } catch (error) {
//     console.error('Registration error:', error);
//     res.status(500).json({ error: 'Server error' });
//   }
// });

router.post('/register', async (req, res) => {
  try {
    console.log("inside register api", req.body);
    const {
      // Authentication fields
      aboId,
      password,
      // Personal Details
      full_name,
      date_of_birth,
      gender,
      ethnicity,
      profile_picture,
      // Communication Details
      mobile_number,
      email,
      address,
      // Identity Details
      national_id,
      identity_proof,
      // Sponsor Details
      sponsor_abo_id,
      sponsor_abo_name,
      // Payment Details
      card_number,
      card_name,
      // Final Verification
      id_proof_image
    }: RegisterCredentials = req.body;

    const existingUser = db.prepare('SELECT * FROM users WHERE abo_id = ?').get(aboId);
    if (existingUser) {
      return res.status(400).json({ error: 'ABO ID already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    
    const result = db.prepare(`
      INSERT INTO users (
        abo_id, password, full_name, date_of_birth, gender, ethnicity, profile_picture,
        mobile_number, email, address, national_id, identity_proof,
        sponsor_abo_id, sponsor_abo_name, card_number, card_name,
        id_proof_image, chat_number
      ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, 
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?
      )
    `).run(
      aboId, hashedPassword, full_name, date_of_birth, gender, ethnicity, profile_picture,
      mobile_number, email, address, national_id, identity_proof,
      sponsor_abo_id, sponsor_abo_name, card_number, card_name,
      id_proof_image, 0
    );

    const token = jwt.sign({ userId: result.lastInsertRowid }, JWT_SECRET, { expiresIn: '24h' });

    // Get the full user data after insertion
    const newUser = db.prepare('SELECT * FROM users WHERE id = ?').get(result.lastInsertRowid);
    
    res.json({
      user: {
        id: newUser.id,
        aboId: newUser.abo_id,
        full_name: newUser.full_name,
        email: newUser.email,
        mobile_number: newUser.mobile_number,
        date_of_birth: newUser.date_of_birth,
        gender: newUser.gender,
        ethnicity: newUser.ethnicity,
        address: newUser.address,
        national_id: newUser.national_id,
        sponsor_abo_id: newUser.sponsor_abo_id,
        sponsor_abo_name: newUser.sponsor_abo_name,
        card_number: newUser.card_number,
        card_name: newUser.card_name,
        createdAt: new Date(newUser.created_at),
        chat_number: newUser.chat_number,
        profile_picture: newUser.profile_picture
      },
      token
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});


router.post('/login', async (req, res) => {
  try {
    const { aboId, password }: LoginCredentials = req.body;
    console.log('All users:', db.prepare('SELECT * FROM users').all());
    const newUser = db.prepare('SELECT * FROM users WHERE abo_id = ?').get(aboId);
    //console.log("after login",user);
    if (!newUser) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const validPassword = await bcrypt.compare(password, newUser.password);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: newUser.id }, JWT_SECRET, { expiresIn: '24h' });
    res.json({
      user: {
        id: newUser.id,
        aboId: newUser.abo_id,
        createdAt: new Date(newUser.created_at),
        chat_number: newUser.chat_number, 
        full_name: newUser.full_name,
        email: newUser.email,
        mobile_number: newUser.mobile_number,
        date_of_birth: newUser.date_of_birth,
        gender: newUser.gender,
        ethnicity: newUser.ethnicity,
        address: newUser.address,
        national_id: newUser.national_id,
        sponsor_abo_id: newUser.sponsor_abo_id,
        sponsor_abo_name: newUser.sponsor_abo_name,
        card_number: newUser.card_number,
        card_name: newUser.card_name,
        profile_picture:newUser.profile_picture
         // Added this line
      },
      token
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

router.get('/validate', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader) {
    return res.status(401).json({ error: 'No token provided' });
  }

  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as { userId: number };
    const user = db.prepare('SELECT * FROM users WHERE id = ?').get(decoded.userId);
    
    if (!user) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    res.json({
      user: {
        id: user.id,
        aboId: user.abo_id,
        createdAt: new Date(user.created_at)
      }
    });
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

export default router;