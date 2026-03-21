from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Claude AI API", description="Python FastAPI + Claude AI 연동 백엔드")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# --- 요청/응답 모델 ---

class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None
    max_tokens: int = 16000


class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int


# --- 엔드포인트 ---

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Claude에게 메시지를 보내고 응답을 반환합니다."""
    try:
        create_kwargs = dict(
            model="claude-opus-4-6",
            max_tokens=req.max_tokens,
            messages=[{"role": "user", "content": req.message}],
        )
        if req.system:
            create_kwargs["system"] = req.system

        response = client.messages.create(**create_kwargs)

        text = next((b.text for b in response.content if b.type == "text"), "")
        return ChatResponse(
            response=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Claude 응답을 스트리밍으로 반환합니다."""

    def generate():
        try:
            create_kwargs = dict(
                model="claude-opus-4-6",
                max_tokens=req.max_tokens,
                messages=[{"role": "user", "content": req.message}],
            )
            if req.system:
                create_kwargs["system"] = req.system

            with client.messages.stream(**create_kwargs) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"

            final = stream.get_final_message()
            yield f"data: {json.dumps({'done': True, 'input_tokens': final.usage.input_tokens, 'output_tokens': final.usage.output_tokens})}\n\n"
        except anthropic.APIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/think")
def chat_think(req: ChatRequest):
    """Claude의 Adaptive Thinking을 활성화해 복잡한 질문에 답합니다."""
    try:
        create_kwargs = dict(
            model="claude-opus-4-6",
            max_tokens=req.max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": req.message}],
        )
        if req.system:
            create_kwargs["system"] = req.system

        response = client.messages.create(**create_kwargs)

        thinking_text = next(
            (b.thinking for b in response.content if b.type == "thinking"), None
        )
        answer_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )

        return {
            "thinking": thinking_text,
            "response": answer_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
