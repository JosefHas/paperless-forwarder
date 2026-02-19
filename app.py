import os, time, json, re, sqlite3, smtplib
from email.message import EmailMessage
from pathlib import Path
import requests
import smtplib
from email.message import EmailMessage

WORKDIR = Path("/work")
WORKDIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WORKDIR / "state.sqlite"

def env(k, d=None):
    v = os.getenv(k, d)
    if v is None:
        raise RuntimeError(f"Missing env var: {k}")
    return v

PAPERLESS_BASE_URL = env("PAPERLESS_BASE_URL", "http://webserver:8000")
PAPERLESS_TOKEN = env("PAPERLESS_TOKEN")

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://ollama:11434")
GATE_MODEL = env("OLLAMA_MODEL_GATE", "qwen2.5:3b")
EXTRACT_MODEL = env("OLLAMA_MODEL_EXTRACT", "llama3.1:8b")
GATE_INVOICE_MIN = float(env("GATE_INVOICE_MIN", "0.35"))
INVOICE_CONFIDENCE_MIN = float(env("INVOICE_CONFIDENCE_MIN", "0.60"))

FARMING_KEYWORDS = [x.strip().lower() for x in env("FARMING_KEYWORDS","").split(",") if x.strip()]
BANK_KEYWORDS = [x.strip().lower() for x in env("BANK_KEYWORDS","").split(",") if x.strip()]
MATCH_IBANS = [re.sub(r"\s+","",x.strip().upper()) for x in env("MATCH_IBANS","").split(",") if x.strip()]

SMTP_HOST = env("SMTP_HOST")
SMTP_PORT = int(env("SMTP_PORT","587"))
SMTP_USER = env("SMTP_USER")
SMTP_PASS = env("SMTP_PASS")
INVOICE_FORWARD_TO = env("INVOICE_FORWARD_TO")

POLL_SECONDS = int(env("POLL_SECONDS","30"))
MAX_DOCS = int(env("MAX_DOCS_PER_LOOP","20"))

IBAN_REGEX = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{0,4}\b")

def db_init():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS processed_docs(
            doc_id INTEGER PRIMARY KEY,
            processed_utc INTEGER NOT NULL
        )""")
        c.commit()

def add_tags_to_document(base, token, doc_id: int, tag_ids: list[int]) -> None:
    h = {"Authorization": f"Token {token}"}

    # Get document
    r = requests.get(f"{base}/api/documents/{doc_id}/", headers=h, timeout=20)
    r.raise_for_status()
    doc = r.json()
    current = set(doc.get("tags", []) or [])
    new = sorted(current.union(set(tag_ids)))

    # Patch document
    r = requests.patch(f"{base}/api/documents/{doc_id}/", headers=h, json={"tags": new}, timeout=20)
    r.raise_for_status()


def get_or_create_tag_id(base, token, tag_name: str) -> int:
    h = {"Authorization": f"Token {token}"}
    # Search existing tags (common DRF filtering)
    r = requests.get(f"{base}/api/tags/", headers=h, params={"name__iexact": tag_name}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", data)  # depending on pagination
    if results:
        return results[0]["id"]

    # Create tag
    r = requests.post(f"{base}/api/tags/", headers=h, json={"name": tag_name}, timeout=20)
    r.raise_for_status()
    return r.json()["id"]

def send_invoice_email(
    smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
    mail_from: str, mail_to: str,
    subject: str, body: str,
    filename: str, pdf_bytes: bytes
):
    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)

def download_pdf_bytes(base, token, doc_id: int, path_template: str) -> bytes:
    h = {"Authorization": f"Token {token}"}
    url = base + path_template.format(id=doc_id)
    r = requests.get(url, headers=h, timeout=120)
    r.raise_for_status()
    return r.content

def already_done(doc_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute("SELECT doc_id FROM processed_docs WHERE doc_id=?", (doc_id,)).fetchone()
        return r is not None

def mark_done(doc_id: int):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR IGNORE INTO processed_docs(doc_id, processed_utc) VALUES(?,?)", (doc_id, int(time.time())))
        c.commit()

def paperless_get_docs():
    url = f"{PAPERLESS_BASE_URL}/api/documents/?ordering=-created&page_size={MAX_DOCS}"
    r = requests.get(url, headers={"Authorization": f"Token {PAPERLESS_TOKEN}"}, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])

def paperless_get_doc_detail(doc_id: int) -> dict:
    url = f"{PAPERLESS_BASE_URL}/api/documents/{doc_id}/"
    r = requests.get(url, headers={"Authorization": f"Token {PAPERLESS_TOKEN}"}, timeout=60)
    r.raise_for_status()
    return r.json()

def paperless_get_text(doc_id: int) -> str:
    # Many Paperless versions expose OCR text as "content" in the document detail.
    d = paperless_get_doc_detail(doc_id)
    return d.get("content") or ""

def ollama_generate(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()
    a, b = out.find("{"), out.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(out[:200])
    return json.loads(out[a:b+1])

def extract_ibans(text: str):
    found=set()
    for m in IBAN_REGEX.finditer(text or ""):
        iban = re.sub(r"\s+","",m.group(0).upper())
        if 15 <= len(iban) <= 34:
            found.add(iban)
    return found

def matches_specific_iban(text: str) -> bool:
    if not MATCH_IBANS: return False
    f = extract_ibans(text)
    return any(x in f for x in MATCH_IBANS)

def contains_any(text: str, kws) -> bool:
    t=(text or "").lower()
    return any(k in t for k in kws if k)

def should_forward(meta: dict, text: str) -> bool:
    if not meta.get("is_invoice"):
        return False
    if float(meta.get("confidence",0.0)) < INVOICE_CONFIDENCE_MIN:
        return False
    farming = (meta.get("job_area")=="farming") or contains_any(text, FARMING_KEYWORDS)
    bank = contains_any(text, BANK_KEYWORDS) or matches_specific_iban(text)
    return farming or bank

def forward_invoice(meta: dict, doc_id: int):
    msg = EmailMessage()
    msg["Subject"] = f"Invoice detected (Paperless #{doc_id}): {meta.get('title','')}"
    msg["From"] = SMTP_USER
    msg["To"] = INVOICE_FORWARD_TO
    msg.set_content(
        f"Invoice detected in Paperless.\n\n"
        f"doc_id: {doc_id}\n"
        f"title: {meta.get('title','')}\n"
        f"invoice_number: {meta.get('invoice_number','')}\n"
        f"total: {meta.get('amount_total','')} {meta.get('currency','')}\n"
        f"date: {meta.get('date','')}\n"
        f"job_area: {meta.get('job_area','')}\n"
        f"notes: {meta.get('notes','')}\n"
    )
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def main():
    db_init()
    while True:
        try:
            docs = paperless_get_docs()
            for d in docs:
                doc_id = int(d["id"])
                if already_done(doc_id):
                    continue

                text = paperless_get_text(doc_id)

                gate = ollama_generate(GATE_MODEL, f"""
Return ONLY JSON: {{"is_invoice":true|false,"confidence":0.0,"job_area":"farming|it|personal|unknown","notes":"short"}}
OCR text:
{text[:6000]}
""".strip())

                run_big = bool(gate.get("is_invoice")) and float(gate.get("confidence",0.0)) >= GATE_INVOICE_MIN

                if run_big:
                    meta = ollama_generate(EXTRACT_MODEL, f"""
Return ONLY JSON:
{{
 "is_invoice":true|false,"confidence":0.0,"job_area":"farming|it|personal|unknown",
 "title":"short","date":"YYYY-MM-DD or empty",
 "amount_total":"string or empty","currency":"string or empty",
 "invoice_number":"string or empty","notes":"short"
}}
OCR text:
{text[:12000]}
""".strip())
                else:
                    meta = {"is_invoice": False, "confidence": float(gate.get("confidence",0.0)), "job_area": gate.get("job_area","unknown")}

                if should_forward(meta, text):
                    forward_invoice(meta, doc_id)

                mark_done(doc_id)

        except Exception as e:
            print("ERROR:", e)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
