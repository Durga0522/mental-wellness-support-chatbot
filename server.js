require("dotenv").config();

const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// PostgreSQL / Neon connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false,
  },
});

// Create tables
async function createTables() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password TEXT NOT NULL,
      role VARCHAR(20) NOT NULL DEFAULT 'user',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS items (
      id SERIAL PRIMARY KEY,
      name VARCHAR(150) NOT NULL,
      description TEXT,
      quantity INT NOT NULL DEFAULT 0,
      price NUMERIC(10,2) NOT NULL DEFAULT 0,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);
}

// JWT token
function generateToken(user) {
  return jwt.sign(
    {
      id: user.id,
      email: user.email,
      role: user.role,
    },
    process.env.JWT_SECRET || "mysecretkey",
    { expiresIn: "1d" }
  );
}

// Auth middleware
function authMiddleware(req, res, next) {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({
        message: "Access denied. No token provided.",
      });
    }

    const token = authHeader.split(" ")[1];
    const decoded = jwt.verify(
      token,
      process.env.JWT_SECRET || "mysecretkey"
    );

    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({
      message: "Invalid or expired token.",
    });
  }
}

// Role middleware
function roleMiddleware(...allowedRoles) {
  return (req, res, next) => {
    if (!req.user || !allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        message: "Access denied. Insufficient permissions.",
      });
    }
    next();
  };
}

// Root route
app.get("/", (req, res) => {
  res.json({
    message: "Inventory Management API is running successfully",
  });
});

// Register
app.post("/api/auth/register", async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({
        message: "Name, email, and password are required",
      });
    }

    if (password.length < 6) {
      return res.status(400).json({
        message: "Password must be at least 6 characters long",
      });
    }

    const existingUser = await pool.query(
      "SELECT * FROM users WHERE email = $1",
      [email]
    );

    if (existingUser.rows.length > 0) {
      return res.status(400).json({
        message: "User already exists",
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = await pool.query(
      `INSERT INTO users (name, email, password, role)
       VALUES ($1, $2, $3, $4)
       RETURNING id, name, email, role, created_at`,
      [name, email, hashedPassword, role || "user"]
    );

    res.status(201).json({
      message: "User registered successfully",
      user: newUser.rows[0],
    });
  } catch (error) {
    console.error("Register error:", error.message);
    res.status(500).json({
      message: "Server error during registration",
    });
  }
});

// Login
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        message: "Email and password are required",
      });
    }

    const userResult = await pool.query(
      "SELECT * FROM users WHERE email = $1",
      [email]
    );

    if (userResult.rows.length === 0) {
      return res.status(400).json({
        message: "Invalid email or password",
      });
    }

    const user = userResult.rows[0];
    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      return res.status(400).json({
        message: "Invalid email or password",
      });
    }

    const token = generateToken(user);

    res.status(200).json({
      message: "Login successful",
      token,
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    console.error("Login error:", error.message);
    res.status(500).json({
      message: "Server error during login",
    });
  }
});

// Get all items
app.get("/api/items", authMiddleware, async (req, res) => {
  try {
    const { search, page = 1, limit = 5 } = req.query;

    const pageNumber = parseInt(page);
    const limitNumber = parseInt(limit);
    const offset = (pageNumber - 1) * limitNumber;

    let query = "SELECT * FROM items";
    let countQuery = "SELECT COUNT(*) FROM items";
    let values = [];

    if (search) {
      query += " WHERE name ILIKE $1";
      countQuery += " WHERE name ILIKE $1";
      values.push(`%${search}%`);
    }

    query += ` ORDER BY id DESC LIMIT $${values.length + 1} OFFSET $${values.length + 2}`;
    values.push(limitNumber, offset);

    const itemsResult = await pool.query(query, values);
    const countValues = search ? [`%${search}%`] : [];
    const totalResult = await pool.query(countQuery, countValues);

    res.status(200).json({
      currentPage: pageNumber,
      totalItems: parseInt(totalResult.rows[0].count),
      itemsPerPage: limitNumber,
      items: itemsResult.rows,
    });
  } catch (error) {
    console.error("Get items error:", error.message);
    res.status(500).json({
      message: "Server error while fetching items",
    });
  }
});

// Get single item
app.get("/api/items/:id", authMiddleware, async (req, res) => {
  try {
    const { id } = req.params;

    const itemResult = await pool.query(
      "SELECT * FROM items WHERE id = $1",
      [id]
    );

    if (itemResult.rows.length === 0) {
      return res.status(404).json({
        message: "Item not found",
      });
    }

    res.status(200).json(itemResult.rows[0]);
  } catch (error) {
    console.error("Get item error:", error.message);
    res.status(500).json({
      message: "Server error while fetching item",
    });
  }
});

// Create item
app.post("/api/items", authMiddleware, roleMiddleware("admin"), async (req, res) => {
  try {
    const { name, description, quantity, price } = req.body;

    if (!name || quantity === undefined || price === undefined) {
      return res.status(400).json({
        message: "Name, quantity, and price are required",
      });
    }

    const newItem = await pool.query(
      `INSERT INTO items (name, description, quantity, price)
       VALUES ($1, $2, $3, $4)
       RETURNING *`,
      [name, description || "", quantity, price]
    );

    res.status(201).json({
      message: "Item created successfully",
      item: newItem.rows[0],
    });
  } catch (error) {
    console.error("Create item error:", error.message);
    res.status(500).json({
      message: "Server error while creating item",
    });
  }
});

// Update item
app.put("/api/items/:id", authMiddleware, roleMiddleware("admin"), async (req, res) => {
  try {
    const { id } = req.params;
    const { name, description, quantity, price } = req.body;

    const existingItem = await pool.query(
      "SELECT * FROM items WHERE id = $1",
      [id]
    );

    if (existingItem.rows.length === 0) {
      return res.status(404).json({
        message: "Item not found",
      });
    }

    const updatedItem = await pool.query(
      `UPDATE items
       SET name = $1,
           description = $2,
           quantity = $3,
           price = $4
       WHERE id = $5
       RETURNING *`,
      [
        name ?? existingItem.rows[0].name,
        description ?? existingItem.rows[0].description,
        quantity ?? existingItem.rows[0].quantity,
        price ?? existingItem.rows[0].price,
        id,
      ]
    );

    res.status(200).json({
      message: "Item updated successfully",
      item: updatedItem.rows[0],
    });
  } catch (error) {
    console.error("Update item error:", error.message);
    res.status(500).json({
      message: "Server error while updating item",
    });
  }
});

// Delete item
app.delete("/api/items/:id", authMiddleware, roleMiddleware("admin"), async (req, res) => {
  try {
    const { id } = req.params;

    const existingItem = await pool.query(
      "SELECT * FROM items WHERE id = $1",
      [id]
    );

    if (existingItem.rows.length === 0) {
      return res.status(404).json({
        message: "Item not found",
      });
    }

    await pool.query("DELETE FROM items WHERE id = $1", [id]);

    res.status(200).json({
      message: "Item deleted successfully",
    });
  } catch (error) {
    console.error("Delete item error:", error.message);
    res.status(500).json({
      message: "Server error while deleting item",
    });
  }
});

// Start server
app.listen(PORT, async () => {
  try {
    await createTables();
    console.log("Tables are ready");
    console.log(`Server running on port ${PORT}`);
  } catch (error) {
    console.error("Startup error:", error.message);
  }
});