from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import uvicorn
import tempfile

app = FastAPI()

#pip install -r requirements.txt
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#Desktop\python
#env\Scripts\activate


# Autoriser l'appel depuis Supabase Edge Function ou local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace par ["https://your-supabase-project.supabase.co"] en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Stocker temporairement le PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Extraction de texte avec PyMuPDF
        doc = fitz.open(tmp_path)
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()

        if not full_text.strip():
            raise HTTPException(status_code=422, detail="No extractable text found in PDF")

        return {
            "filename": file.filename,
            "text": full_text,
            "page_count": len(doc),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")

