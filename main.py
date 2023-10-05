import io

from fastapi import FastAPI, UploadFile
from PIL import Image

from pytorch_model import model_pipeline

app = FastAPI()


@app.get('/')
def read_root() -> str:
    return "Theres onnly one endpoint - POST /image_upload"


@app.post('/image_upload')
def image_upload(text: str, image: UploadFile) -> dict[str, str]:
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    
    return {"answer": model_pipeline(text, image)}
