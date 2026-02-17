import asyncio
import logging
import uuid
import base64
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import prompts
from services.stt_factory import STTServiceFactory
from services.tts_factory import TTSServiceFactory
from services.llm_service import GroqLLMService
from services.telephony_factory import TelephonyServiceFactory
from services.enquiry_storage import EnquiryStorage
from services.knowledge_validator import KnowledgeValidator

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Property Enquiry Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
active_sessions = {}
storage = EnquiryStorage(config.ENQUIRIES_FILE)
telephony_service = None

# Dynamic fields for Brigade Eternia site visit flow
BRIGADE_ETERNIA_DYNAMIC_FIELDS = {
    "conversation_stage": {
        "type": "string",
        "description": "Stage",
        "default": "identity_check"
    },
    "preferred_bhk": {
        "type": "string",
        "description": "BHK",
        "default": "none"
    },
    "visit_date": {
        "type": "string",
        "description": "Date",
        "default": "none"
    },
    "visit_time": {
        "type": "string",
        "description": "Time",
        "default": "none"
    },
    "visit_confirmed": {
        "type": "string",
        "description": "Confirmed",
        "default": "no"
    }
}

class EnquirySubmission(BaseModel):
    name: str
    phone: str
    email: str
    message: str = ""

@app.on_event("startup")
async def startup():
    global telephony_service
    
    logger.info("=" * 80)
    logger.info("BRIGADE ETERNIA VOICE AGENT")
    logger.info(f"Project: {config.PROJECT_NAME}")
    logger.info(f"Agent: {config.AGENT_NAME}")
    logger.info(f"Telephony: Exotel")
    logger.info("=" * 80)
    
    # Initialize telephony service
    telephony_service = TelephonyServiceFactory.create(
        provider=config.TELEPHONY_PROVIDER,
        account_sid=config.EXOTEL_ACCOUNT_SID,
        api_key=config.EXOTEL_API_KEY,
        api_token=config.EXOTEL_API_TOKEN,
        subdomain=config.EXOTEL_SUBDOMAIN,
        webhook_url=config.WEBHOOK_BASE_URL
    )
    
    logger.info(f"STT: {config.STT_PROVIDER}, TTS: {config.TTS_PROVIDER}")
    logger.info(f"Call delay: {config.CALL_DELAY_SECONDS}s")
    logger.info("=" * 80)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/submit-enquiry")
async def submit_enquiry(enquiry: EnquirySubmission):
    enquiry_id = str(uuid.uuid4())
    
    enquiry_data = {
        "enquiry_id": enquiry_id,
        "form_data": {
            "name": enquiry.name,
            "phone": enquiry.phone,
            "email": enquiry.email,
            "message": enquiry.message
        },
        "submitted_at": datetime.now().isoformat(),
        "call_data": None,
        "status": "pending"
    }
    
    await storage.save_enquiry(enquiry_data)
    logger.info(f"Enquiry submitted: {enquiry_id} - {enquiry.name}")
    
    # Schedule call
    asyncio.create_task(schedule_call(enquiry_id, enquiry.phone))
    
    return {"status": "success", "enquiry_id": enquiry_id, "message": "Call scheduled"}

async def schedule_call(enquiry_id: str, phone: str):
    logger.info(f"Scheduling call for {enquiry_id} in {config.CALL_DELAY_SECONDS}s")
    await asyncio.sleep(config.CALL_DELAY_SECONDS)
    
    logger.info(f"Initiating call to {phone}")
    # Exotel Strategy: Call Customer (From) -> Connect to Virtual Number (To) which holds the Applet/Flow
    result = await telephony_service.make_call(
        from_number=phone,
        to_number=config.EXOTEL_PHONE_NUMBER,
        session_id=enquiry_id
    )
    
    if result["status"] == "success":
        await storage.update_enquiry(enquiry_id, {
            "status": "calling",
            "call_sid": result.get("call_uuid")
        })
        logger.info(f"Call initiated: {result.get('call_uuid')}")
    else:
        logger.error(f"Call failed: {result.get('message')}")
        await storage.update_enquiry(enquiry_id, {"status": "failed"})

@app.api_route("/exotel-webhook", methods=["GET", "POST"])
async def exotel_webhook(request: Request):
    session_id = request.query_params.get("session_id")
    logger.info(f"Exotel webhook for session: {session_id}")
    
    # Just log the event as we use Applet for logic
    try:
        data = await request.json()
        logger.info(f"Webhook Payload: {data}")
    except:
        body = await request.body()
        logger.info(f"Webhook Body: {body.decode()}")
    
    return JSONResponse(content={"status": "ok"})

@app.websocket("/exotel_stream")
async def exotel_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    
    try:
        # Get session_id from first message or headers
        message = await websocket.receive_text()
        data = json.loads(message)
        session_id = data.get("session_id") or data.get("headers", {}).get("session_id")
        
        logger.info(f"WebSocket connected: {session_id}")
        
        # Initialize session
        await initialize_call_session(session_id, websocket)
        
        # Handle audio stream
        async for message in websocket.iter_text():
            data = json.loads(message)
            
            if data.get("event") == "media":
                audio_chunk = base64.b64decode(data["media"]["payload"])
                session = active_sessions.get(session_id)
                if session:
                    await session["stt_service"].process_audio(audio_chunk)
            
            elif data.get("event") == "stop":
                await cleanup_session(session_id)
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await websocket.close()

async def initialize_call_session(session_id: str, websocket: WebSocket):
    logger.info(f"Initializing session: {session_id}")
    
    # Get enquiry data
    enquiry = await storage.get_enquiry(session_id)
    if not enquiry:
        logger.error(f"Enquiry not found: {session_id}")
        return
    
    form_data = enquiry["form_data"]
    
    # Initialize services
    stt_api_key = (
        config.DEEPGRAM_API_KEY if config.STT_PROVIDER == 'deepgram'
        else config.SARVAM_API_KEY
    )
    tts_api_key = (
        config.CARTESIA_API_KEY if config.TTS_PROVIDER == 'cartesia'
        else config.SARVAM_API_KEY
    )
    
    stt_service = STTServiceFactory.create(
        provider=config.STT_PROVIDER,
        api_key=stt_api_key
    )
    
    tts_kwargs = {
        'language': config.TTS_LANGUAGE,
        'speed': config.CARTESIA_SPEED if config.TTS_PROVIDER == 'cartesia' else config.SARVAM_SPEED
    }
    if config.TTS_PROVIDER == 'cartesia':
        tts_kwargs['model_id'] = config.CARTESIA_MODEL_ID
    
    tts_service = TTSServiceFactory.create(
        provider=config.TTS_PROVIDER,
        api_key=tts_api_key,
        voice_id=config.VOICE_ID,
        **tts_kwargs
    )
    
    llm_service = GroqLLMService(api_key=config.GROQ_API_KEY, max_history=config.LLM_MAX_HISTORY)
    
    # Initialize all services
    await stt_service.initialize(
        api_key=stt_api_key,
        callback=lambda text: handle_transcription(text, session_id)
    )
    await tts_service.initialize()
    
    # Get system prompt
    system_prompt = prompts.get_formatted_prompt(
        user_name=form_data["name"],
        user_message=form_data["message"]
    )
    
    await llm_service.initialize(
        dynamic_fields=BRIGADE_ETERNIA_DYNAMIC_FIELDS,
        system_prompt_template=system_prompt
    )
    
    active_sessions[session_id] = {
        "session_id": session_id,
        "websocket": websocket,
        "stt_service": stt_service,
        "tts_service": tts_service,
        "llm_service": llm_service,
        "conversation_history": [],
        "start_time": datetime.now(),
        "enquiry_data": enquiry,
        "completed_stages": [],       # Track what's done
        "current_stage": "identity_check",  # Track current stage
        "call_ended": False
    }
    
    # Send greeting with first name only
    first_name = form_data['name'].strip().split()[0] if form_data['name'] else "there"
    
    # Use centralized greeting from config
    greeting = config.GREETING_TEMPLATE.format(name=first_name)
    
    await tts_service.synthesize(
        text=greeting,
        send_audio_callback=lambda chunk, action: send_audio_to_exotel(websocket, chunk, action)
    )

async def handle_transcription(text: str, session_id: str):
    if text == "__FORCE_STOP__":
        session = active_sessions.get(session_id)
        if session:
            await session["tts_service"].stop()
        return
    
    session = active_sessions.get(session_id)
    if not session:
        return
    
    # If call already ended, ignore everything
    if session.get("call_ended"):
        logger.info(f"[{session_id}] Call already ended, ignoring input")
        return
    
    logger.info(f"[{session_id}] User: {text}")
    
    # Stop TTS
    await session["tts_service"].stop()
    
    # Add to history
    session["conversation_history"].append({"role": "user", "content": text})
    
    # Build stage context to inject
    completed_stages = session.get("completed_stages", [])
    stage_context = ""
    if completed_stages:
        stage_context = (
            f"\n[SYSTEM NOTE: Completed stages: {', '.join(completed_stages)}. "
            f"Current stage: {session.get('current_stage', 'unknown')}. "
            f"FORBIDDEN: Do NOT ask 'Am I speaking with {session.get('user_name', 'user')}?' - identity already confirmed. "
            f"After answering user's question, return ONLY to current stage question.]"
        )
    # Inject stage context into user input
    augmented_input = text
    if stage_context:
        augmented_input = text + stage_context

    # Get LLM response
    response = await session["llm_service"].generate_response(
        user_input=augmented_input,
        conversation_history=session["conversation_history"]
    )
    
    ai_text = response["response"].response
    collected_data = response["raw_model_data"]

    # Update stage tracking based on AI response
    ai_lower = ai_text.lower()

    # Detect which stage just completed based on response content
    if session["current_stage"] == "identity_check" and any(p in ai_lower for p in ["rohan from brigade", "good time to talk"]):
        if "identity_check" not in session["completed_stages"]:
            session["completed_stages"].append("identity_check")
        session["current_stage"] = "timing_check"

    elif session["current_stage"] == "timing_check" and any(p in ai_lower for p in ["3 bhk or 4 bhk", "looking for 3"]):
        if "timing_check" not in session["completed_stages"]:
            session["completed_stages"].append("timing_check")
            session["completed_stages"].append("intro_said")  # Mark intro as done
        session["current_stage"] = "bhk_preference"

    elif session["current_stage"] == "bhk_preference" and "how soon" in ai_lower:
        if "bhk_preference" not in session["completed_stages"]:
            session["completed_stages"].append("bhk_preference")
        session["current_stage"] = "urgency_assessment"

    elif session["current_stage"] == "urgency_assessment" and any(p in ai_lower for p in ["visit", "site", "weekend"]):
        if "urgency_assessment" not in session["completed_stages"]:
            session["completed_stages"].append("urgency_assessment")
        session["current_stage"] = "site_visit_scheduling"

    elif "confirmed" in ai_lower and any(p in ai_lower for p in ["visit confirmed", "whatsapp confirmation"]):
        session["current_stage"] = "post_visit_confirmed"
        session["call_ended_after_farewell"] = False

    logger.info(f"[{session_id}] Stage: {session['current_stage']} | Completed: {session['completed_stages']}")

    # Fix: Force farewell if user says no more questions
    no_more_questions_phrases = [
        "no more questions", "no questions", "that's all", "thats all",
        "nothing else", "i'm good", "im good", "nope", "no thanks", "no"
    ]
    farewell_phrases = ["have a wonderful day", "have a great day", "pleasure speaking"]

    user_done = any(phrase in text.lower() for phrase in no_more_questions_phrases)
    ai_has_farewell = any(phrase in ai_text.lower() for phrase in farewell_phrases)

    if user_done and not ai_has_farewell:
        ai_text = "Great! I'll share the floor plans and location on WhatsApp. Pleasure speaking with you. Have a wonderful day!"
        logger.info(f"[{session_id}] Forced farewell message")
        session["call_ended"] = True
    
    # Log model usage
    model_used = response.get("model_used", "unknown")
    was_fallback = response.get("was_fallback", False)
    model_info = f" (Model: {model_used})"
    if was_fallback:
        model_info += " [FALLBACK]"
    
    logger.info(f"[{session_id}] AI{model_info}: {ai_text}")
    logger.info(f"[{session_id}] Collected: {collected_data}")
    
    # Check if site visit booked
    if collected_data.get("visit_date") != "none" and collected_data.get("visit_time") != "none":
        logger.info(f"[{session_id}] Site visit booked: {collected_data['visit_date']} at {collected_data['visit_time']}")
        
        # Mark as success
        await storage.update_enquiry(session_id, {
            "status": "site_visit_booked",
            "visit_scheduled": {
                "date": collected_data["visit_date"],
                "time": collected_data["visit_time"]
            },
            "call_data": {
                "collected_info": collected_data,
                "conversation_history": session["conversation_history"]
            }
        })
        
        # This will be the final message, end call after TTS
        session["ending_soon"] = True
    
    # Add to history
    session["conversation_history"].append({"role": "assistant", "content": ai_text})
    
    # Save collected data (if not already saved above)
    if not session.get("ending_soon"):
        await storage.update_enquiry(session_id, {
            "call_data": {
                "collected_info": collected_data,
                "conversation_history": session["conversation_history"]
            }
        })
    
    # Set call_ended flag BEFORE synthesizing if this is farewell
    current_response_is_farewell = any(phrase in ai_text.lower() for phrase in farewell_phrases)

    if session.get("ending_soon") or response["should_end_call"] or current_response_is_farewell:
        session["call_ended"] = True
        logger.info(f"[{session_id}] Call ending - stopping STT immediately")
        
        # STOP STT NOW so no more transcriptions come in
        try:
            await session["stt_service"].close()
            logger.info(f"[{session_id}] STT stopped successfully")
        except Exception as e:
            logger.warning(f"[{session_id}] STT stop error: {e}")

    # Synthesize response
    await session["tts_service"].synthesize(...)

    # End call after TTS completes
    if session.get("call_ended"):
        await asyncio.sleep(1)
        await cleanup_session(session_id)

async def send_audio_to_exotel(websocket: WebSocket, audio_chunk, action: str):
    if action == "playAudio" and audio_chunk:
        await websocket.send_json({
            "event": "media",
            "media": {
                "payload": base64.b64encode(audio_chunk).decode()
            }
        })
    elif action == "clearAudio":
        await websocket.send_json({"event": "clear"})

async def cleanup_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        return
    
    logger.info(f"Cleaning up session: {session_id}")
    
    # Close services
    await session["stt_service"].close()
    await session["tts_service"].close()
    await session["llm_service"].close()
    
    # Calculate duration
    duration = (datetime.now() - session["start_time"]).total_seconds()
    
    # Update storage
    await storage.update_enquiry(session_id, {
        "status": "completed",
        "call_data": {
            **session["enquiry_data"].get("call_data", {}),
            "duration": duration,
            "ended_at": datetime.now().isoformat()
        }
    })
    
    del active_sessions[session_id]
    logger.info(f"Session cleaned: {session_id}")

@app.get("/enquiries")
async def get_enquiries():
    enquiries = await storage.get_all_enquiries()
    return {"enquiries": enquiries}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "provider": config.TELEPHONY_PROVIDER
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
