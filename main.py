from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#from rag_pipeline import chat_with_csv
import os


app = FastAPI(title='RAG Backend + Frontend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

UPLOAD_FOLDER = 'uploads'
FRONTEND_FOLDER = 'frontend'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount('/frontend', StaticFiles(directory=FRONTEND_FOLDER), name='frontend')
templates = Jinja2Templates(directory=FRONTEND_FOLDER)

current_csv_path = None


@app.get('/', response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


# @app.post('/upload-csv')
# async def upload_csv(csv_file: UploadFile = File(...)):
#     global current_csv_path

#     file_location = os.path.join(UPLOAD_FOLDER, csv_file.filename)
#     with open(file_location, 'wb') as f:
#         f.write(await csv_file.read())

#     current_csv_path = file_location
#     return {'message': f"📄 CSV '{csv_file.filename}' uploaded successfully."}


# @app.post('/ask')
# async def ask_question(payload: dict):
#     global current_csv_path

#     user_query = payload.get('query', '')

#     if not current_csv_path:
#         return {'response': '⚠️ Please upload a CSV file first.'}

#     with open(current_csv_path, 'rb') as csv_file:
#         response = chat_with_csv(csv_file, user_query)

#     return {'response': response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
