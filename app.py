# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
import binascii

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.50  # Lowered threshold for better recall
MAX_RESULTS = 10  # Increased to get more context
load_dotenv()
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per source
API_KEY = os.getenv("API_KEY")  # Get API key from environment variable

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    
    # Create markdown_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call the embedding API through aipipe proxy
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database
async def find_similar_content(query_embedding, max_results=MAX_RESULTS):
    """Find similar content from both discourse and markdown chunks"""
    similar_chunks = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Search discourse chunks
        cursor.execute("SELECT * FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_rows = cursor.fetchall()
        
        for row in discourse_rows:
            try:
                stored_embedding = json.loads(row['embedding'])
                similarity = cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    similar_chunks.append({
                        'content': row['content'],
                        'similarity': similarity,
                        'source': 'discourse',
                        'url': row['url'],
                        'topic_title': row['topic_title'],
                        'author': row['author'],
                        'created_at': row['created_at'],
                        'post_number': row['post_number'],
                        'topic_id': row['topic_id'],
                        'post_id': row['post_id'],
                        'chunk_index': row['chunk_index']
                    })
            except Exception as e:
                logger.error(f"Error processing discourse row {row['id']}: {e}")
                continue
        
        # Search markdown chunks
        cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_rows = cursor.fetchall()
        
        for row in markdown_rows:
            try:
                stored_embedding = json.loads(row['embedding'])
                similarity = cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    similar_chunks.append({
                        'content': row['content'],
                        'similarity': similarity,
                        'source': 'markdown',
                        'url': row['original_url'],
                        'doc_title': row['doc_title'],
                        'chunk_index': row['chunk_index']
                    })
            except Exception as e:
                logger.error(f"Error processing markdown row {row['id']}: {e}")
                continue
                
        conn.close()
        
        # Sort by similarity and return top results
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_chunks[:max_results]
        
    except Exception as e:
        logger.error(f"Error finding similar content: {e}")
        logger.error(traceback.format_exc())
        return []

# Function to enrich results with adjacent chunks for better context
async def enrich_with_adjacent_chunks(similar_chunks):
    """Add adjacent chunks for better context"""
    enriched_chunks = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for chunk in similar_chunks:
            enriched_chunk = chunk.copy()
            
            if chunk['source'] == 'discourse':
                # Get adjacent chunks from the same post
                cursor.execute("""
                    SELECT content, chunk_index FROM discourse_chunks 
                    WHERE post_id = ? AND topic_id = ? 
                    ORDER BY chunk_index
                """, (chunk['post_id'], chunk['topic_id']))
                
                adjacent_chunks = cursor.fetchall()
                current_index = chunk['chunk_index']
                
                # Collect surrounding chunks
                context_chunks = []
                for adj_chunk in adjacent_chunks:
                    if abs(adj_chunk['chunk_index'] - current_index) <= MAX_CONTEXT_CHUNKS:
                        context_chunks.append(adj_chunk['content'])
                
                enriched_chunk['full_context'] = ' '.join(context_chunks)
                
            elif chunk['source'] == 'markdown':
                # Get adjacent chunks from the same document
                cursor.execute("""
                    SELECT content, chunk_index FROM markdown_chunks 
                    WHERE doc_title = ? 
                    ORDER BY chunk_index
                """, (chunk['doc_title'],))
                
                adjacent_chunks = cursor.fetchall()
                current_index = chunk['chunk_index']
                
                # Collect surrounding chunks
                context_chunks = []
                for adj_chunk in adjacent_chunks:
                    if abs(adj_chunk['chunk_index'] - current_index) <= MAX_CONTEXT_CHUNKS:
                        context_chunks.append(adj_chunk['content'])
                
                enriched_chunk['full_context'] = ' '.join(context_chunks)
            
            enriched_chunks.append(enriched_chunk)
        
        conn.close()
        return enriched_chunks
        
    except Exception as e:
        logger.error(f"Error enriching chunks: {e}")
        logger.error(traceback.format_exc())
        return similar_chunks  # Return original chunks if enrichment fails

# Function to call LLM for answer generation
async def generate_answer(question, context_chunks, image_base64=None):
    """Generate answer using LLM with context"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured")

    try:
        # Prepare context from chunks
        context_parts = []
        links = []

        for chunk in context_chunks:
            if chunk['source'] == 'discourse':
                context_text = f"From Discourse post '{chunk['topic_title']}' by {chunk['author']}:\n{chunk.get('full_context', chunk['content'])}"
                link_text = f"{chunk['topic_title']} - Post #{chunk['post_number']}"
            else:
                context_text = f"From document '{chunk['doc_title']}':\n{chunk.get('full_context', chunk['content'])}"
                link_text = chunk['doc_title']

            context_parts.append(context_text)

            if chunk['url'] and not any(link['url'] == chunk['url'] for link in links):
                links.append({
                    'url': chunk['url'],
                    'text': link_text[:100] + "..." if len(link_text) > 100 else link_text
                })

        context = "\n\n---\n\n".join(context_parts)

        # Base prompt
        messages = [
            {
                "role": "system",
                "content": """You are a helpful teaching assistant for the Tools in Data Science course at IIT Madras. 
                Answer student questions based on the provided context from course materials and discourse discussions.

                Guidelines:
                - Be concise and direct
                - Reference specific information from the context when relevant
                - If the context doesn't contain enough information to answer the question, say so
                - For technical questions, provide clear step-by-step guidance
                - Maintain a helpful and encouraging tone"""
            }
        ]

        user_content = f"Question: {question}\n\nContext from course materials:\n{context}"

        # Optional image
        if image_base64 and isinstance(image_base64, str):
            try:
                base64.b64decode(image_base64, validate=True)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            except Exception as e:
                logger.warning(f"Skipping image due to base64 error: {e}")
                messages.append({
                    "role": "user",
                    "content": user_content
                })
        else:
            messages.append({
                "role": "user",
                "content": user_content
            })

        # LLM call
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result["choices"][0]["message"]["content"]
                    return answer, links
                else:
                    error_text = await response.text()
                    logger.error(f"Error calling LLM API (status {response.status}): {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"LLM API error: {error_text}")

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/api/")
async def query_knowledge_base(request: Request):
    """Main endpoint for querying the knowledge base - handles raw stringified JSON from promptfoo"""
    try:
        body = await request.body()

        # Parse JSON even if sent as string
        try:
            body_text = body.decode('utf-8')
            request_data = json.loads(body_text)
        except Exception as e:
            logger.error(f"JSON decode error: {e}")
            return {
                "data": {
                    "answer": "I couldn't parse the request. Please check the format.",
                    "links": []
                }
            }

        # Extract inputs
        question = request_data.get("question")
        image = request_data.get("image")

        if not question:
            return {
                "data": {
                    "answer": "The request must include a 'question' field.",
                    "links": []
                }
            }

        logger.info(f"Received query: {question[:100]}...")

        # Generate embedding
        try:
            query_embedding = await get_embedding(question)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return {
                "data": {
                    "answer": "There was a problem generating the embedding.",
                    "links": []
                }
            }

        # Search similar content
        try:
            similar_chunks = await find_similar_content(query_embedding)
            logger.info(f"Found {len(similar_chunks)} similar chunks")
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return {
                "data": {
                    "answer": "I couldn't search the knowledge base properly.",
                    "links": []
                }
            }

        # Always proceed, even if empty context
        if not similar_chunks:
            logger.warning("No strong matches found, proceeding with empty context")
            similar_chunks = []

        # Enrich with context
        try:
            enriched_chunks = await enrich_with_adjacent_chunks(similar_chunks)
        except Exception as e:
            logger.error(f"Enrichment error: {e}")
            enriched_chunks = similar_chunks

        # Generate final answer
        try:
            answer, links = await generate_answer(question, enriched_chunks, image)
            return {
                "data": {
                    "answer": answer or "I couldn't generate a full answer.",
                    "links": links or []
                }
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "data": {
                    "answer": "There was a problem generating the final answer.",
                    "links": []
                }
            }

    except Exception as e:
        logger.error(f"Top-level exception: {e}")
        return {
            "data": {
                "answer": "An unexpected error occurred while processing your request.",
                "links": []
            }
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Try to connect to the database as part of health check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)