import os
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator

import tiktoken
import redis.asyncio as redis_async  # Changed to async Redis client
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse  # Added StreamingResponse import
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain and Vector Store Imports
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Removed other model providers to focus only on OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Rate limiting
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("NyayaGPT-API")

# === API Models ===
class QueryRequest(BaseModel):
    query: str
    model_name: str = "gpt-3.5-turbo"  # Default to faster model
    conversation_id: Optional[str] = None
    strategy: str = "simple"  # Default to simpler strategy for speed
    max_tokens: int = 512  # Reduced token limit for faster responses
    temperature: float = 0.1  # Lower temperature for more deterministic responses
    stream: bool = True  # Enable streaming by default for faster perceived response

class ResponseMetadata(BaseModel):
    model: str
    strategy: str
    chunks_retrieved: int
    tokens_used: int
    processing_time: float
    conversation_id: str

class QueryResponse(BaseModel):
    response: str
    metadata: ResponseMetadata
    context_sources: List[Dict[str, str]] = []

class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: List[str]

class BulkDeleteRequest(BaseModel):
    message_indices: List[int]

# === Redis Configuration ===
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL = int(os.getenv("REDIS_TTL", 60 * 60 * 24 * 7))  # Default 7 days
CACHE_TTL = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # Cache responses for 24 hours

# === Initialize App ===
app = FastAPI(
    title="NyayaGPT API",
    description="Legal Assistant API powered by LLMs with RAG",
    version="1.0.0"
)

# === Add CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Redis ===
async def init_redis():
    redis_instance = redis_async.from_url(REDIS_URL, decode_responses=True)
    await FastAPILimiter.init(redis_instance)
    return redis_instance

redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    try:
        redis_client = await init_redis()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        # Continue without Redis - features requiring Redis will be disabled
    
    try:
        # Initialize vector store
        init_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise  # This is critical, so we should fail startup

# === Initialize Vector Store ===
vector_store = None

def init_vector_store():
    global vector_store
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("Pinecone API key not found")
        raise ValueError("Pinecone API key is required")
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "assistant-index"
    
    if not pc.has_index(index_name):
        logger.error(f"Pinecone index '{index_name}' does not exist")
        raise ValueError(f"Pinecone index '{index_name}' does not exist")
    
    index = pc.Index(index_name)
    # Use lite embedding model for faster embedding generation
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002")  # Using faster embedding model
    )

# === LLM Configuration ===
AVAILABLE_MODELS = {
    "gpt-4o": lambda streaming=False: ChatOpenAI(
        model="gpt-4o", 
        temperature=0.1, 
        max_tokens=512,
        streaming=streaming,
        request_timeout=20  # 20 seconds timeout
    ),
    "gpt-4o-mini": lambda streaming=False: ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1, 
        max_tokens=512,
        streaming=streaming,
        request_timeout=15  # 15 seconds timeout 
    ),
    "gpt-3.5-turbo": lambda streaming=False: ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.1, 
        max_tokens=512,
        streaming=streaming,
        request_timeout=10  # 10 seconds timeout
    )
}

# === Prompt Templates ===
final_prompt = PromptTemplate(
    template="""
You are "NyayaGPT", a professional legal assistant trained on Supreme Court and High Court judgments, Indian statutes, procedural codes, and legal drafting standards.

Your task is to:
1. Understand the user's fact pattern or legal query
2. Identify applicable laws, leading judgments, and strategic points.
3. Suggest winning arguments based on precedent when asked to do so only.
4. Draft legal documents such as petitions, applications, notices, replies, or complaints (template style only) when asked to do so only.
5. Provide list of cases separately as per the query.
6. Provide some details of cases mentioned.
Never provide personal legal advice or guaranteed outcomes. Always include citations, issue-based structure, and disclaimers. Proceed step-by-step:

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

fusion_prompt = ChatPromptTemplate.from_template("""
You are an assistant skilled in legal language modeling.
Given the following user query, generate 3 different rephrasings of it as formal Indian legal questions.
Do not invent extra facts or foreign law. Just reword using Indian legal terminology.

User Query: {question}

Three Rephrasings:""")

# === Utility Functions ===
def format_docs(docs, max_length=800):
    """Format documents with a more efficient length limit to avoid token wastage"""
    result = []
    for doc in docs:
        title = doc.metadata.get("title", "Untitled Document")
        url = doc.metadata.get("url", "No URL")
        # Use shorter excerpts to save tokens
        result.append(f"### {title}\n**Source:** {url}\n\n{doc.page_content.strip()[:max_length]}...")
    return "\n\n".join(result)

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# === Retrieval Strategies ===
def fusion_strategy(query, llm):
    """Simplified fusion strategy with fewer queries and documents"""
    # Generate query variations - only 2 instead of 3
    fusion_chain = fusion_prompt | llm
    response = fusion_chain.invoke({"question": query})
    variants = [line.strip("- ") for line in response.content.strip().split("\n") if line.strip()][:2]
    variants.insert(0, query)  # Add the original query
    
    # Retrieve documents for each variant, but fewer per variant
    seen = set()
    all_docs = []
    
    # Use asyncio gather for concurrent retrieval
    for variant in variants:
        for doc in vector_store.similarity_search(variant, k=1):  # Only 1 doc per variant
            hash_ = doc.page_content[:100]
            if hash_ not in seen:
                seen.add(hash_)
                all_docs.append(doc)
    
    return all_docs[:3]  # Return max 3 documents

def simple_strategy(query, llm):
    """Direct retrieval with fewer documents"""
    return vector_store.similarity_search(query, k=2)  # Only 2 docs instead of 4

# === Redis Conversation Storage ===
async def get_conversation(conversation_id):
    if not redis_client:
        logger.error("Redis client not initialized")
        return []
    
    conversation_data = await redis_client.get(f"conv:{conversation_id}")
    if conversation_data:
        return json.loads(conversation_data)
    return []

async def save_conversation(conversation_id, query, response):
    if not redis_client:
        logger.error("Redis client not initialized")
        return
    
    # Get existing conversation or create new one
    conversation = await get_conversation(conversation_id)
    
    # Add new message pair
    conversation.append({
        "timestamp": time.time(),
        "query": query,
        "response": response
    })
    
    # Save conversation with TTL
    await redis_client.setex(
        f"conv:{conversation_id}", 
        REDIS_TTL, 
        json.dumps(conversation)
    )

# === Cache Helper Functions ===
async def get_cached_response(query: str, model_name: str, strategy: str):
    """Get cached response if available"""
    if not redis_client:
        return None
    
    cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
    cached = await redis_client.get(cache_key)
    
    if cached:
        logger.info(f"Cache hit for query: {query[:30]}...")
        return json.loads(cached)
    return None

async def cache_response(query: str, model_name: str, strategy: str, response_data: dict):
    """Cache response for future use"""
    if not redis_client:
        return
    
    cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
    await redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(response_data)
    )
    logger.info(f"Cached response for query: {query[:30]}...")

# === Streaming Response Generator ===
async def generate_streaming_response(query_request: QueryRequest, background_tasks: BackgroundTasks) -> AsyncGenerator[str, None]:
    """Generate a streaming response for the query."""
    start_time = time.time()
    
    # Initialize conversation ID if not provided
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Check if model is available
        if query_request.model_name not in AVAILABLE_MODELS:
            error_msg = json.dumps({
                "error": f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            })
            yield f"data: {error_msg}\n\n"
            return
        
        # Initialize LLM with streaming option
        llm = AVAILABLE_MODELS[query_request.model_name](streaming=True)
        llm.temperature = query_request.temperature
        llm.max_tokens = query_request.max_tokens
        
        # Select retrieval strategy
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        # Retrieve relevant documents (with timeout protection)
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
        
        # Format documents and create context (with shorter max length for speed)
        context = format_docs(docs, max_length=600)
        
        # Create prompt
        prompt = final_prompt.format(
            context=context, 
            question=query_request.query
        )
        
        # Count tokens
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Create chain with callbacks for streaming
        chain = llm | StrOutputParser()
        
        # Stream the response
        full_response = ""
        async for chunk in chain.astream(prompt):
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
        
        # Save to conversation history in the background
        background_tasks.add_task(
            save_conversation,
            conversation_id,
            query_request.query,
            full_response
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create sources list (with shorter snippets)
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", "No URL"),
                "snippet": doc.page_content[:150] + "..."  # Shorter snippets
            }
            for doc in docs
        ]
        
        # Send completion metadata
        completion_data = {
            "done": True,
            "metadata": {
                "model": query_request.model_name,
                "strategy": query_request.strategy,
                "chunks_retrieved": len(docs),
                "tokens_used": tokens_used,
                "processing_time": round(duration, 2),
                "conversation_id": conversation_id
            },
            "context_sources": sources
        }
        
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# === Core Query Processing ===
async def process_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks
):
    start_time = time.time()
    
    # Initialize conversation ID if not provided
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Check cache first for identical queries (if not streaming)
        if not query_request.stream:
            cached = await get_cached_response(
                query_request.query, 
                query_request.model_name,
                query_request.strategy
            )
            if cached:
                # Update conversation ID in cached response
                cached["metadata"]["conversation_id"] = conversation_id
                
                # Save to conversation history
                await save_conversation(
                    conversation_id, 
                    query_request.query, 
                    cached["response"]
                )
                
                # Return cached response with updated conversation ID
                return QueryResponse(**cached)
        
        # Get LLM model
        if query_request.model_name not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Initialize LLM with streaming option if requested
        llm = AVAILABLE_MODELS[query_request.model_name](streaming=query_request.stream)
        llm.temperature = query_request.temperature
        llm.max_tokens = query_request.max_tokens
        
        # Select retrieval strategy
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        # Retrieve relevant documents (with timeout protection)
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
        
        # Format documents and create context (with shorter max length for speed)
        context = format_docs(docs, max_length=600)
        
        # Create prompt
        prompt = final_prompt.format(
            context=context, 
            question=query_request.query
        )
        
        # Count tokens
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Generate response
        parser = StrOutputParser()
        answer = (llm | parser).invoke(prompt)
        
        # Create sources list (with shorter snippets)
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", "No URL"),
                "snippet": doc.page_content[:150] + "..."  # Shorter snippets
            }
            for doc in docs
        ]
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create response
        response = QueryResponse(
            response=answer,
            metadata=ResponseMetadata(
                model=query_request.model_name,
                strategy=query_request.strategy,
                chunks_retrieved=len(docs),
                tokens_used=tokens_used,
                processing_time=round(duration, 2),
                conversation_id=conversation_id
            ),
            context_sources=sources
        )
        
        # Save to conversation history
        await save_conversation(conversation_id, query_request.query, answer)
        
        # Cache response if not streaming
        if not query_request.stream:
            await cache_response(
                query_request.query,
                query_request.model_name,
                query_request.strategy,
                response.dict()
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# === API Endpoints ===
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available models"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/clear-cache")
async def clear_cache():
    """Clear the response cache"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    # Get all cache keys
    cursor = 0
    deleted_count = 0
    
    while True:
        cursor, keys = await redis_client.scan(cursor, match="cache:*")
        if keys:
            deleted = await redis_client.delete(*keys)
            deleted_count += deleted
        
        if cursor == 0:
            break
    
    return {
        "status": "success",
        "message": f"Cache cleared: {deleted_count} entries removed"
    }

# === Generate First-Time Conversation ID ===
async def get_or_create_conversation(request: Request) -> str:
    """Get existing conversation ID from cookie or create a new one"""
    # Try to get conversation_id from cookie
    conversation_id = request.cookies.get("conversation_id")
    
    if not conversation_id:
        # Generate new ID if none exists
        conversation_id = str(uuid.uuid4())
        logger.info(f"Created new conversation: {conversation_id}")
    
    return conversation_id

@app.post(
    "/query", 
    dependencies=[Depends(RateLimiter(times=20, seconds=60))]  # Increased rate limit for better responsiveness
)
async def query_endpoint(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Process a legal query using the specified LLM and retrieval strategy"""
    # Auto-generate conversation ID if not provided
    if not query_request.conversation_id:
        query_request.conversation_id = await get_or_create_conversation(request)
    
    # If streaming is requested, return a streaming response
    if query_request.stream:
        response = StreamingResponse(
            generate_streaming_response(query_request, background_tasks),
            media_type="text/event-stream"
        )
        
        # Set conversation ID cookie in response
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60  # 30 days
        )
        
        return response
    
    # Regular non-streaming response
    response_data = await process_query(query_request, background_tasks)
    
    # Return the response with cookie
    response = JSONResponse(content=response_data.dict())
    response.set_cookie(
        key="conversation_id",
        value=query_request.conversation_id,
        httponly=True,
        max_age=30*24*60*60  # 30 days
    )
    
    return response

@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history by ID"""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    return {"conversation_id": conversation_id, "messages": conversation}

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation by ID"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    deleted = await redis_client.delete(f"conv:{conversation_id}")
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    
    return {"status": "success", "message": f"Conversation {conversation_id} deleted"}

@app.delete("/conversation/{conversation_id}/message/{message_index}")
async def delete_message(conversation_id: str, message_index: int):
    """Delete a specific message from a conversation by its index"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    # Get the conversation
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    
    # Check if message index is valid
    if message_index < 0 or message_index >= len(conversation):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid message index {message_index}"
        )
    
    # Remove the message
    removed_message = conversation.pop(message_index)
    
    # Save the updated conversation
    await redis_client.setex(
        f"conv:{conversation_id}", 
        REDIS_TTL, 
        json.dumps(conversation)
    )
    
    return {
        "status": "success", 
        "message": f"Message at index {message_index} deleted",
        "removed": removed_message
    }

@app.delete("/conversation/{conversation_id}/messages")
async def delete_multiple_messages(conversation_id: str, request: BulkDeleteRequest):
    """Delete multiple messages from a conversation by their indices"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    # Get the conversation
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    
    # Sort indices in descending order to avoid index shifting issues
    indices = sorted(request.message_indices, reverse=True)
    
    # Check if indices are valid
    if any(idx < 0 or idx >= len(conversation) for idx in indices):
        raise HTTPException(
            status_code=400,
            detail="One or more invalid message indices"
        )
    
    # Remove messages
    removed = []
    for idx in indices:
        removed.append(conversation.pop(idx))
    
    # Save the updated conversation
    await redis_client.setex(
        f"conv:{conversation_id}", 
        REDIS_TTL, 
        json.dumps(conversation)
    )
    
    return {
        "status": "success", 
        "message": f"Deleted {len(indices)} messages",
        "removed": removed
    }

# === Exception Handlers ===
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )

# === Main Application Entry ===
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )