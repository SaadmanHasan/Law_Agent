import os
import io
import csv
from pathlib import Path

from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import easyocr
from dotenv import load_dotenv

from text_recog import messages_from_ocr
from ingestion import rebuild_vectorstore, CHAT_CSV, DOCS_DIR
from qa_chain import answer_question

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER_IMAGES"] = "uploads/images"
app.config["UPLOAD_FOLDER_DOCS"] = "uploads/docs"

os.makedirs(app.config["UPLOAD_FOLDER_IMAGES"], exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER_DOCS"], exist_ok=True)

reader = easyocr.Reader(["en", "ch_sim"])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    files = request.files.getlist("images")
    all_rows = []

    for f in files:
        if not f.filename:
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER_IMAGES"], filename)
        f.save(save_path)

        img = Image.open(save_path)
        img_width, img_height = img.size
        ocr_result = reader.readtext(save_path, detail=1)

        rows = messages_from_ocr(
            ocr_result,
            image_name=Path(filename).name,
            img_height=img_height,
            img_width=img_width,
        )
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["Date"], r["Time"]))

    DATA_DIR = CHAT_CSV.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with CHAT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Date", "Time", "Sender", "Message", "Source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
    output.seek(0)

    rebuild_vectorstore()

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="chat_history.csv",
    )

@app.route("/qa", methods=["GET"])
def qa_page():
    return render_template("qa.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    result = answer_question(question)
    return jsonify(result)

@app.route("/upload_docs", methods=["POST"])
def upload_docs():
    files = request.files.getlist("docs")
    saved = []
    for f in files:
        if not f.filename:
            continue
        filename = secure_filename(f.filename)
        path = Path(app.config["UPLOAD_FOLDER_DOCS"]) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        f.save(path)
        saved.append(filename)

    
    rebuild_vectorstore()

    return jsonify({"uploaded": saved})

if __name__ == "__main__":
    app.run(debug=True)
