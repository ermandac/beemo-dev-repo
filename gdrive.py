from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = 'C:/Users/Carl Joseph Torres/Desktop/BeemoDosDist/BeemoApp/GsheetAPI/beemo-2-main-443903-3075ec515af5.json'

def authenticate_drive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def upload_to_drive(file_path, folder_id, drive_service):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]  # Specify the folder ID here
    }
    media = MediaFileUpload(file_path, mimetype='image/png')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File ID: {file.get('id')}")