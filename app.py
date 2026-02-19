import os, time, json, re, sqlite3, smtplib
from pathlib import Path
from email.message import EmailMessage
import requests

WORKDIR = Path("/work")
WORKDIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WORKDIR / "state.sqlite"

def env(k, d=None):
    v = os.getenv(k, d)
    if v is None:
        raise RuntimeError(f"Missing env var: {k}")
    return v

# ---------------------------
# Config
# ---------------------------
PAPERLESS_BASE_URL = env("PAPERLESS_BASE_URL", "http://webserver:8000").rstrip("/")
PAPERLESS_TOKEN = env("PAPERLESS_TOKEN")
PAPERLESS_DOWNLOAD_PATH_TEMPLATE = env("PAPERLESS_DOWNLOAD_PATH_TEMPLATE", "/api/documents/{id}/download/")

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
GATE_MODEL = env("OLLAMA_MODEL_GATE", "qwen2.5:3b")
EXTRACT_MODEL = env("OLLAMA_MODEL_EXTRACT", "llama3.1:8b")
OLLAMA_AUTO_PULL = env("OLLAMA_AUTO_PULL", "1") == "1"
OLLAMA_PRUNE_UNUSED = env("OLLAMA_PRUNE_UNUSED", "0") == "1"

GATE_INVOICE_MIN = float(env("GATE_INVOICE_MIN", "0.35"))
INVOICE_CONFIDENCE_MIN = float(env("INVOICE_CONFIDENCE_MIN", "0.60"))

MATCH_IBANS = [re.sub(r"\s+","",x.strip().upper()) for x in env("MATCH_IBANS","").split(",") if x.strip()]

SMTP_HOST = env("SMTP_HOST")
SMTP_PORT = int(env("SMTP_PORT","587"))
SMTP_USER = env("SMTP_USER")
SMTP_PASS = env("SMTP_PASS")
MAIL_FROM = env("MAIL_FROM", SMTP_USER)
INVOICE_FORWARD_TO = env("INVOICE_FORWARD_TO")

POLL_SECONDS = int(env("POLL_SECONDS","30"))
MAX_DOCS = int(env("MAX_DOCS_PER_LOOP","20"))

# Tags (created if missing)
TAG_INVOICE = env("TAG_INVOICE", "invoice")
TAG_FORWARDED = env("TAG_FORWARDED", "invoice-forwarded")
TAG_NOT_FORWARDED = env("TAG_NOT_FORWARDED", "invoice-not-forwarded")
TAG_REASON_FARM = env("TAG_REASON_FARM", "farm")
TAG_REASON_IBAN = env("TAG_REASON_IBAN", "iban-match")

IBAN_REGEX = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{0,4}\b")

def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "-", *a, flush=True)

# ---------------------------
# DB
# ---------------------------
def db_init():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS processed_docs(
            doc_id INTEGER PRIMARY KEY,
            processed_utc INTEGER NOT NULL
        )""")
        c.commit()

def already_done(doc_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute("SELECT doc_id FROM processed_docs WHERE doc_id=?", (doc_id,)).fetchone()
        return r is not None

def mark_done(doc_id: int):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR IGNORE INTO processed_docs(doc_id, processed_utc) VALUES(?,?)",
                  (doc_id, int(time.time())))
        c.commit()

# ---------------------------
# Paperless API
# ---------------------------
def paperless_headers():
    return {"Authorization": f"Token {PAPERLESS_TOKEN}"}

def paperless_get_docs():
    # newest first; page_size supported on recent versions
    url = f"{PAPERLESS_BASE_URL}/api/documents/"
    params = {"ordering": "-created", "page_size": MAX_DOCS}
    r = requests.get(url, headers=paperless_headers(), params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("results", data)

def paperless_get_doc_detail(doc_id: int) -> dict:
    url = f"{PAPERLESS_BASE_URL}/api/documents/{doc_id}/"
    r = requests.get(url, headers=paperless_headers(), timeout=60)
    r.raise_for_status()
    return r.json()

def paperless_get_text(doc_id: int) -> str:
    d = paperless_get_doc_detail(doc_id)
    return d.get("content") or ""

def download_pdf_bytes(doc_id: int) -> bytes:
    url = PAPERLESS_BASE_URL + PAPERLESS_DOWNLOAD_PATH_TEMPLATE.format(id=doc_id)
    r = requests.get(url, headers=paperless_headers(), timeout=180)
    r.raise_for_status()
    return r.content

# ---------------------------
# Tagging
# ---------------------------
def get_or_create_tag_id(tag_name: str) -> int:
    h = paperless_headers()
    r = requests.get(f"{PAPERLESS_BASE_URL}/api/tags/", headers=h, params={"name__iexact": tag_name}, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", data)
    if results:
        return int(results[0]["id"])

    r = requests.post(f"{PAPERLESS_BASE_URL}/api/tags/", headers=h, json={"name": tag_name}, timeout=30)
    r.raise_for_status()
    return int(r.json()["id"])

def add_tags_to_document(doc_id: int, tag_ids: list[int]) -> None:
    h = paperless_headers()
    r = requests.get(f"{PAPERLESS_BASE_URL}/api/documents/{doc_id}/", headers=h, timeout=30)
    r.raise_for_status()
    doc = r.json()
    current = set(doc.get("tags", []) or [])
    new = sorted(current.union(set(tag_ids)))
    r = requests.patch(f"{PAPERLESS_BASE_URL}/api/documents/{doc_id}/", headers=h, json={"tags": new}, timeout=30)
    r.raise_for_status()

# ---------------------------
# Ollama
# ---------------------------
def ollama_tags() -> set[str]:
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    r.raise_for_status()
    models = r.json().get("models", [])
    return set(m.get("name") for m in models if m.get("name"))

def ollama_pull(model: str):
    log("Pulling model:", model)
    r = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model, "stream": False}, timeout=3600)
    r.raise_for_status()

def ollama_delete(model: str):
    log("Deleting model:", model)
    r = requests.delete(f"{OLLAMA_BASE_URL}/api/delete", json={"name": model}, timeout=120)
    r.raise_for_status()

def ensure_models():
    try:
        have = ollama_tags()
    except Exception as e:
        log("ERROR: cannot reach Ollama /api/tags:", e)
        return

    needed = {GATE_MODEL, EXTRACT_MODEL}
    if OLLAMA_AUTO_PULL:
        for m in needed:
            if m not in have:
                try:
                    ollama_pull(m)
                except Exception as e:
                    log("ERROR pulling model", m, ":", e)

    if OLLAMA_PRUNE_UNUSED:
        try:
            have2 = ollama_tags()
            for m in sorted(have2):
                if m not in needed:
                    try:
                        ollama_delete(m)
                    except Exception as e:
                        log("ERROR deleting model", m, ":", e)
        except Exception as e:
            log("ERROR during prune:", e)

def ollama_generate(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()

    # try to parse JSON blob from response
    a, b = out.find("{"), out.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(f"Model did not return JSON. First 200 chars: {out[:200]}")
    return json.loads(out[a:b+1])

# ---------------------------
# Rules
# ---------------------------
def extract_ibans(text: str) -> set[str]:
    found=set()
    for m in IBAN_REGEX.finditer(text or ""):
        iban = re.sub(r"\s+","",m.group(0).upper())
        if 15 <= len(iban) <= 34:
            found.add(iban)
    return found

def matches_specific_iban(text: str) -> bool:
    if not MATCH_IBANS:
        return False
    found = extract_ibans(text)
    return any(x in found for x in MATCH_IBANS)

def should_forward(meta: dict, text: str) -> tuple[bool, dict]:
    """
    Returns (forward?, reasons dict)
    reasons: {"iban": bool, "farm": bool}
    """
    if not meta.get("is_invoice"):
        return False, {"iban": False, "farm": False}

    conf = float(meta.get("invoice_confidence", meta.get("confidence", 0.0)) or 0.0)
    if conf < INVOICE_CONFIDENCE_MIN:
        return False, {"iban": False, "farm": False}

    iban_match = matches_specific_iban(text)
    farm = bool(meta.get("is_farming_related")) and float(meta.get("farming_confidence", 0.0) or 0.0) >= 0.5

    return (iban_match or farm), {"iban": iban_match, "farm": farm}

# ---------------------------
# Email
# ---------------------------
def send_invoice_email(subject: str, body: str, filename: str, pdf_bytes: bytes):
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = INVOICE_FORWARD_TO
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# ---------------------------
# Main
# ---------------------------
def main():
    db_init()
    log("Forwarder started")
    log("Paperless:", PAPERLESS_BASE_URL)
    log("Ollama:", OLLAMA_BASE_URL)
    ensure_models()

    # Create tags once
    tag_ids = {
        "invoice": get_or_create_tag_id(TAG_INVOICE),
        "forwarded": get_or_create_tag_id(TAG_FORWARDED),
        "not_forwarded": get_or_create_tag_id(TAG_NOT_FORWARDED),
        "farm": get_or_create_tag_id(TAG_REASON_FARM),
        "iban": get_or_create_tag_id(TAG_REASON_IBAN),
    }

    while True:
        try:
            docs = paperless_get_docs()
            log(f"Polling... got {len(docs)} docs (latest)")

            for d in docs:
                doc_id = int(d["id"])
                if already_done(doc_id):
                    continue

                # Wait for OCR content; don't mark done if empty
                text = paperless_get_text(doc_id)
                if not text.strip():
                    log(f"Doc {doc_id}: OCR text empty -> skip for now")
                    continue

                # Gate: invoice + farming signal
                gate_prompt = f"""
Return ONLY JSON:
{{
  "is_invoice": true|false,
  "confidence": 0.0,
  "is_farming_related": true|false,
  "farming_confidence": 0.0,
  "notes": "short"
}}
OCR text:
{text[:6000]}
""".strip()

                gate = ollama_generate(GATE_MODEL, gate_prompt)
                gate_is_invoice = bool(gate.get("is_invoice"))
                gate_conf = float(gate.get("confidence", 0.0) or 0.0)

                run_big = gate_is_invoice and gate_conf >= GATE_INVOICE_MIN

                if run_big:
                    extract_prompt = f"""
Return ONLY JSON:
{{
  "is_invoice": true|false,
  "invoice_confidence": 0.0,
  "is_farming_related": true|false,
  "farming_confidence": 0.0,

  "title": "short",
  "date": "YYYY-MM-DD or empty",
  "amount_total": "string or empty",
  "currency": "string or empty",
  "invoice_number": "string or empty",
  "notes": "short"
}}
OCR text:
{text[:12000]}
""".strip()
                    meta = ollama_generate(EXTRACT_MODEL, extract_prompt)
                else:
                    meta = {
                        "is_invoice": False,
                        "invoice_confidence": gate_conf,
                        "is_farming_related": bool(gate.get("is_farming_related")),
                        "farming_confidence": float(gate.get("farming_confidence", 0.0) or 0.0),
                        "notes": gate.get("notes", "")
                    }

                # Tag if invoice (even if not forwarded)
                is_invoice = bool(meta.get("is_invoice"))
                reasons = {"iban": False, "farm": False}

                if is_invoice:
                    add_tags_to_document(doc_id, [tag_ids["invoice"]])

                forward, reasons = should_forward(meta, text)

                if forward:
                    # Download PDF and email with attachment
                    pdf = download_pdf_bytes(doc_id)
                    filename = f"paperless_{doc_id}.pdf"

                    subject = f"Invoice (Paperless #{doc_id}): {meta.get('title','')}".strip()
                    body = (
                        "Invoice forwarded from Paperless.\n\n"
                        f"doc_id: {doc_id}\n"
                        f"title: {meta.get('title','')}\n"
                        f"invoice_number: {meta.get('invoice_number','')}\n"
                        f"total: {meta.get('amount_total','')} {meta.get('currency','')}\n"
                        f"date: {meta.get('date','')}\n"
                        f"invoice_confidence: {meta.get('invoice_confidence', meta.get('confidence',''))}\n"
                        f"farming_related: {meta.get('is_farming_related','')}\n"
                        f"farming_confidence: {meta.get('farming_confidence','')}\n"
                        f"iban_match: {reasons.get('iban')}\n"
                        f"notes: {meta.get('notes','')}\n"
                    )

                    send_invoice_email(subject, body, filename, pdf)

                    # Tag forwarded + reasons
                    to_add = [tag_ids["forwarded"]]
                    if reasons.get("farm"):
                        to_add.append(tag_ids["farm"])
                    if reasons.get("iban"):
                        to_add.append(tag_ids["iban"])
                    add_tags_to_document(doc_id, to_add)

                    log(f"Doc {doc_id}: forwarded (farm={reasons.get('farm')}, iban={reasons.get('iban')})")
                else:
                    # If it's an invoice but not forwarded, tag that too
                    if is_invoice:
                        add_tags_to_document(doc_id, [tag_ids["not_forwarded"]])
                    log(f"Doc {doc_id}: not forwarded (invoice={is_invoice})")

                mark_done(doc_id)

        except Exception as e:
            log("ERROR:", repr(e))

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
