@app.post('/upload-csv')
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

