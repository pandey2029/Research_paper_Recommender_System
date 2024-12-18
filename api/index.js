
import express from 'express';
//Object Data Modelling library for MongoDB and NodeJS
import mongoose from 'mongoose';
//.env loads environmental variable from .env to process.env
import dotenv from 'dotenv';
import userRoutes from './routes/user.route.js';
import authRoutes from './routes/auth.route.js';
import cookieParser from 'cookie-parser';
//Axios is an upgraded version of fetch. in fetch we give input and get a respoonse and then convert it into json format but in axios, data is by default in json format
//Helps is making HTTP requests from any JS library
import axios from 'axios';
//Cross origin resource sharing
import cors from 'cors';

dotenv.config();

mongoose
  .connect(process.env.MONGO)
  .then(() => {
    console.log('Connected to MongoDB');
  })
  .catch((err) => {
    console.log(err);
  });

const app = express();

// Allow CORS
app.use(cors());

// This is used to receive POST request from the client
app.use(express.json());

app.use(cookieParser());

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});

app.use('/api/user', userRoutes);
app.use('/api/auth', authRoutes);

// Handle research paper submission
app.post('/api/research', async (req, res) => {
  console.log('Request received at /api/research', req.body);
  try {
    const { paperTitle } = req.body;
    const response = await axios.post('http://127.0.0.1:6000/recommend', { paper_title: paperTitle });
    console.log('Response from Flask backend:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error forwarding request to Flask backend:', error);
    res.status(500).json({ error: 'Error forwarding request to Flask backend' });
  }
});

// Fetch paper details from Semantic Scholar
app.get('/api/papers', async (req, res) => {
  const query = req.query.query;

  if (!query) {
    return res.status(400).send('Query parameter is required');
  }

  try {
    // to get paper Id
    const searchResponse = await axios.get(`https://api.semanticscholar.org/graph/v1/paper/search?query=${query}`, {
      headers: { 'x-api-key': process.env.S2_API_KEY }
    });
    const papers = searchResponse.data.data;

    if (papers.length === 0) {
      return res.status(404).send('No papers found for the given query');
    }

    const paperId = papers[0].paperId;

    // To get paper details
    const detailsResponse = await axios.get(`https://api.semanticscholar.org/graph/v1/paper/${paperId}?fields=url,abstract,authors`, {
      headers: { 'x-api-key': process.env.S2_API_KEY }
    });
    const paperDetails = detailsResponse.data;

    res.json(paperDetails);
  } catch (err) {
    res.status(500).send('Error: ' + err.message);
  }
});

// Status code 404- User not found
// Status code 403- User Credentials

//Middleware function for handeling errors

app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const message = err.message || 'Internal Server Error';
  return res.status(statusCode).json({
    success: false,
    message,
    statusCode,
  });
});

