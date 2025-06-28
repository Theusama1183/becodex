from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from datetime import datetime, timedelta
import os
import uuid
import logging
import time
import asyncio
import httpx
import stripe
from jose import jwt, jwk, JWTError
from jose.utils import base64url_decode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration using Pydantic Settings
class Settings(BaseSettings):
    mongo_uri: str = Field(..., env="MONGO_URI")
    clerk_jwks_url: str = Field(..., env="CLERK_JWKS_URL")
    clerk_api_key: str = Field(..., env="CLERK_API_KEY")
    stripe_secret: str = Field(..., env="STRIPE_SECRET")
    stripe_webhook_secret: str = Field(..., env="STRIPE_WEBHOOK_SECRET")
    model_id: str = Field("deepseek-ai/deepseek-coder-1.3b-instruct", env="MODEL_ID")
    pro_price_id: str = Field(..., env="STRIPE_PRO_PRICE_ID")
    debug: bool = Field(False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BeeCodex")

# Initialize FastAPI
app = FastAPI(title="BeeCodex API", version="1.0.0", root_path="/api")

# Security Middleware
if not settings.debug:
    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.yourdomain.com", "localhost"]  # Add your production domain
    )

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup MongoDB
client = AsyncIOMotorClient(settings.mongo_uri)
db = client.beecodex
users_collection = db.users
events_collection = db.events

# Rate limiting configuration
RATE_LIMITS = {"free": 100, "pro": 1000}
RATE_PERIOD = 60  # in seconds

# Clerk JWT verification
jwks_cache = {}
jwks_last_updated = 0

async def fetch_jwks():
    global jwks_cache, jwks_last_updated
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(settings.clerk_jwks_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch JWKS")
        jwks_cache = response.json()
        jwks_last_updated = time.time()
        logger.info("JWKS cache updated")

async def get_clerk_public_key(kid):
    global jwks_cache, jwks_last_updated
    
    # Refresh JWKS if older than 1 hour
    if time.time() - jwks_last_updated > 3600:
        await fetch_jwks()
    
    key_data = next((key for key in jwks_cache.get('keys', []) if key['kid'] == kid), None)
    if not key_data:
        await fetch_jwks()  # Try once more
        key_data = next((key for key in jwks_cache.get('keys', []) if key['kid'] == kid), None)
        if not key_data:
            raise HTTPException(status_code=403, detail="Invalid token key ID")

    return jwk.construct(key_data)

async def verify_clerk_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    try:
        headers = jwt.get_unverified_headers(token)
        key = await get_clerk_public_key(headers["kid"])
        
        claims = jwt.decode(
            token,
            key,
            algorithms=[headers["alg"]],
            options={"require_exp": True, "require_iat": True}
        )
        user_id = claims.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="Invalid token")

        # Check token expiration
        exp = claims.get("exp")
        if exp and datetime.utcnow() > datetime.utcfromtimestamp(exp):
            raise HTTPException(status_code=401, detail="Token expired")

        user = await users_collection.find_one({"clerk_user_id": user_id})
        if not user:
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.get(
                    f"https://api.clerk.dev/v1/users/{user_id}",
                    headers={"Authorization": f"Bearer {settings.clerk_api_key}"}
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to fetch user info from Clerk")
                user_data = res.json()

            user = {
                "clerk_user_id": user_id,
                "email": user_data.get("email_addresses", [{}])[0].get("email_address", ""),
                "username": user_data.get("username"),
                "full_name": user_data.get("first_name", "") + " " + user_data.get("last_name", ""),
                "image_url": user_data.get("image_url"),
                "plan": "free",
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=7),
                "api_key": str(uuid.uuid4()),
                "last_reset": datetime.utcnow(),
                "request_count": 0
            }
            await users_collection.insert_one(user)

        return user
    except JWTError as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=403, detail="Token verification failed")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(200, gt=0, le=1000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, gt=0, le=100)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(1.0, ge=0.0, le=2.0)

# Model loading at startup
@app.on_event("startup")
async def startup_event():
    # Load ML model
    logger.info("Loading ML model...")
    app.state.tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    logger.info("Model loaded successfully")
    
    # Create database indexes
    logger.info("Creating database indexes...")
    await users_collection.create_index("clerk_user_id", unique=True)
    await users_collection.create_index("api_key", unique=True)
    await events_collection.create_index("event_id", unique=True)
    logger.info("Database indexes created")
    
    # Initial JWKS fetch
    await fetch_jwks()
    
    # Start JWKS refresh task
    asyncio.create_task(jwks_refresh_task())

async def jwks_refresh_task():
    while True:
        await asyncio.sleep(3600)  # Refresh every hour
        try:
            await fetch_jwks()
        except Exception as e:
            logger.error(f"JWKS refresh failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Welcome to BeeCodex API"}

@app.post("/completion")
async def completion(prompt: PromptRequest, authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")

    scheme, _, api_key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not api_key:
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    user = await users_collection.find_one({"api_key": api_key})
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Check plan expiration
    if user["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=403, detail="Plan expired")

    # Check rate limits
    plan = user.get("plan", "free")
    limit = RATE_LIMITS.get(plan, 100)
    
    # Reset counter if period has passed
    now = datetime.utcnow()
    reset_time = user["last_reset"] + timedelta(seconds=RATE_PERIOD)
    if now > reset_time:
        update = {"$set": {"last_reset": now, "request_count": 1}}
    else:
        if user["request_count"] >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        update = {"$inc": {"request_count": 1}}
    
    # Update usage atomically
    result = await users_collection.update_one(
        {"_id": user["_id"], "request_count": user["request_count"]},
        update
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=429, detail="Concurrent rate limit exceeded")

    # Generate completion
    inputs = app.state.tokenizer(prompt.prompt, return_tensors="pt").to(app.state.model.device)
    with torch.no_grad():
        output = app.state.model.generate(
            **inputs,
            max_new_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            top_k=prompt.top_k,
            top_p=prompt.top_p,
            repetition_penalty=prompt.repetition_penalty,
            do_sample=True,
            pad_token_id=app.state.tokenizer.eos_token_id
        )
    decoded_output = app.state.tokenizer.decode(output[0], skip_special_tokens=True)
    completion_text = decoded_output[len(prompt.prompt):].strip()

    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "prompt": prompt.prompt,
        "completion": completion_text
    }

@app.post("/regenerate-api-key")
async def regenerate_api_key(user=Depends(verify_clerk_token)):
    new_key = str(uuid.uuid4())
    await users_collection.update_one(
        {"clerk_user_id": user["clerk_user_id"]},
        {"$set": {"api_key": new_key}}
    )
    return {"message": "API key regenerated successfully", "api_key": new_key}

@app.get("/usage")
async def usage(user=Depends(verify_clerk_token)):
    now = datetime.utcnow()
    reset_time = user["last_reset"] + timedelta(seconds=RATE_PERIOD)
    remaining = RATE_LIMITS.get(user.get("plan", "free"), 100) - user["request_count"]
    
    return {
        "clerk_user_id": user['clerk_user_id'],
        "plan": user.get("plan", "free"),
        "requests_made": user["request_count"],
        "rate_limit": RATE_LIMITS.get(user.get("plan", "free"), 100),
        "remaining_requests_in_window": max(0, remaining),
        "window_reset": reset_time.isoformat(),
        "total_usage": user.get("total_usage", 0)
    }

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    stripe.api_key = settings.stripe_secret
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            settings.stripe_webhook_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Check for duplicate events
    event_id = event.get("id")
    if await events_collection.find_one({"event_id": event_id}):
        return {"status": "duplicate"}

    # Process event
    try:
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = session.get("client_reference_id")
            if not user_id:
                logger.error("Missing user ID in Stripe session")
                return {"status": "missing_user_id"}
                
            # Get price details
            line_items = stripe.checkout.Session.list_line_items(session["id"], limit=1)
            if not line_items or len(line_items.data) == 0:
                logger.error("No line items in session")
                return {"status": "missing_line_items"}
                
            price_id = line_items.data[0].price.id
            
            # Determine plan duration
            if price_id == settings.pro_price_id:
                duration = timedelta(days=365)
            else:
                duration = timedelta(days=30)
                
            # Update user plan
            result = await users_collection.update_one(
                {"clerk_user_id": user_id},
                {
                    "$set": {
                        "plan": "pro",
                        "expires_at": datetime.utcnow() + duration,
                        "last_reset": datetime.utcnow(),
                        "request_count": 0
                    }
                }
            )
            
            if result.modified_count == 0:
                logger.warning(f"User not found: {user_id}")
        
        # Record processed event
        await events_collection.insert_one({"event_id": event_id, "processed_at": datetime.utcnow()})
        
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing failed")

    return {"status": "success"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "db_connected": await db.command("ping") is not None,
        "jwks_valid": jwks_last_updated > 0
    }