from fastapi import FastAPI, Request, Response
from api.endpoints import handle_chat_request

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    query = await request.json()
    result = handle_chat_request(query)
    return Response(content=result, media_type="application/json")
