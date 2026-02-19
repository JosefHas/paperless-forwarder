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

def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "-", *a, flush=True)

# ---------------------------
# Config
# ---------------------------
PAPERLESS_BASE_URL = env("PAPERLESS_BASE_URL", "http://webserver:8000").rstrip("/")
PAPERLESS_TOKEN = env("PAPERLESS_TOKEN")
PAPERLESS_DOWNLOAD_PATH_TEMPLATE = env("PAPERLESS_DOWNLOAD_PATH_TEMPLATE", "/api/documents/{id}/download/")

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
GATE_MODEL = env("OLLAMA_MODEL_GATE", "qwen2.5:3b")
EXTRACT_MODEL = env("OLLAMA_MODEL_EXTRACT", "llama3.1:8b")
GATE_INVOICE_MIN = float(env("GATE_INVOICE_MIN", "0.35"))
INVOICE_CONFIDENCE_MIN = float(env("INVOICE_CONFIDENCE_MIN", "0.60"))

# keywords enabled again
KEYWORDS_FARMING = [x.strip().lower() for x in env("KEYWORDS_FARMING", "").split(",") if x.strip()]
KEYWORDS_IT = [x.strip().lower() for x in env("KEYWORDS_IT", "").split(",") if x.strip()]

MATCH_IBANS = [re.sub(r"\s+","",x.strip().upper()) for x in env("MATCH_IBANS","").split(",") if x.strip()]

SMTP_HOST = env("SMTP_HOST")
SMTP_PORT = int(env("SMTP_PORT","587"))
SMTP_USER = env("SMTP_USER")
SMTP_PASS = env("SMTP_PASS")
MAIL_FROM = env("MAIL_FROM", SMTP_USER)

FARM_FORWARD_TO = env("FARM_FORWARD_TO")
IT_FORWARD_TO = env("IT_FORWARD_TO")

POLL_SECONDS = int(env("POLL_SECONDS","30"))
MAX_DOCS = int(env("MAX_DOCS_PER_LOOP","20"))

# Tags (created if missing)
TAG_AI = env("TAG_AI", "ai-processed")
TAG_INVOICE = env("TAG_INVOICE", "invoice")
TAG_FORWARDED = env("TAG_FORWARDED", "invoice-forwarded")
TAG_NOT_FORWARDED = env("TAG_NOT_FORWARDED", "invoice-not-forwarded")

TAG_REASON_IBAN = env("TAG_REASON_IBAN", "reason-iban")
TAG_REASON_KEYWORD = env("TAG_REASON_KEYWORD", "reason-keyword")
TAG_TOPIC_FARM = env("TAG_TOPIC_FARM", "topic-farm")
TAG_TOPIC_IT = env("TAG_TOPIC_IT", "topic-it")
TAG_IT_DEDUCTIBLE = env("TAG_IT_DEDUCTIBLE", "deductible-it")

IBAN_REGEX = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{0,4}\b")

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
def ollama_generate(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()
    a, b = out.find("{"), out.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(f"Model did not return JSON. First 200 chars: {out[:200]}")
    return json.loads(out[a:b+1])

def contains_any(text: str, kws) -> bool:
    t = (text or "").lower()
    return any(k in t for k in kws if k)

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

# ---------------------------
# Email
# ---------------------------
def send_email_with_pdf(to_addr: str, subject: str, body: str, filename: str, pdf_bytes: bytes):
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# ---------------------------
# Startup waits
# ---------------------------
def wait_for_paperless():
    url = f"{PAPERLESS_BASE_URL}/api/"
    for _ in range(90):
        try:
            r = requests.get(url, headers=paperless_headers(), timeout=5)
            if r.status_code in (200, 401, 403):
                log("Paperless reachable:", r.status_code)
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Paperless not reachable after waiting")

def wait_for_ollama():
    url = f"{OLLAMA_BASE_URL}/api/tags"
    for _ in range(90):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                log("Ollama reachable")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Ollama not reachable after waiting")

# ---------------------------
# Decision logic
# ---------------------------
def decide_and_route(meta: dict, text: str) -> dict:
    """
    Returns a dict with:
      invoice(bool), invoice_conf(float),
      iban_match(bool), keyword_match(bool),
      ai_farm(bool), ai_it(bool),
      it_deductible(bool),
      forward_farm(bool), forward_it(bool),
      reasons(list[str])
    """
    invoice = bool(meta.get("is_invoice"))
    inv_conf = float(meta.get("invoice_confidence", meta.get("confidence", 0.0)) or 0.0)

    iban_match = matches_specific_iban(text)
    keyword_match = contains_any(text, KEYWORDS_FARMING) or contains_any(text, KEYWORDS_IT)

    ai_farm = bool(meta.get("is_farming_related")) and float(meta.get("farming_confidence", 0.0) or 0.0) >= 0.5
    ai_it = bool(meta.get("is_it_related")) and float(meta.get("it_confidence", 0.0) or 0.0) >= 0.5

    # "German tax write-off" is not legal advice; we treat this as a practical heuristic classification:
    # only forward to IT mailbox when AI believes it's a business IT expense (not personal).
    it_deductible = bool(meta.get("it_deductible_for_tax")) and ai_it

    reasons = []
    if iban_match: reasons.append("iban")
    if keyword_match: reasons.append("keyword")
    if ai_farm: reasons.append("ai_farm")
    if ai_it: reasons.append("ai_it")
    if it_deductible: reasons.append("it_deductible")

    # gate: must be invoice + confidence
    if (not invoice) or inv_conf < INVOICE_CONFIDENCE_MIN:
        return {
            "invoice": invoice, "invoice_conf": inv_conf,
            "iban_match": iban_match, "keyword_match": keyword_match,
            "ai_farm": ai_farm, "ai_it": ai_it, "it_deductible": it_deductible,
            "forward_farm": False, "forward_it": False,
            "reasons": reasons,
        }

    # forward rule you asked:
    # invoice AND (iban OR keywords OR ai farming OR ai says IT & deductible)
    triggers = iban_match or keyword_match or ai_farm or it_deductible

    forward_farm = triggers and (ai_farm or contains_any(text, KEYWORDS_FARMING))
    forward_it = triggers and (it_deductible or contains_any(text, KEYWORDS_IT))

    # If neither topic matches but triggers via IBAN/keyword, default to farm mailbox (safe default)
    if triggers and (not forward_farm) and (not forward_it):
        forward_farm = True
        reasons.append("default_farm")

    return {
        "invoice": invoice, "invoice_conf": inv_conf,
        "iban_match": iban_match, "keyword_match": keyword_match,
        "ai_farm": ai_farm, "ai_it": ai_it, "it_deductible": it_deductible,
        "forward_farm": forward_farm, "forward_it": forward_it,
        "reasons": reasons,
    }

# ---------------------------
# Main
# ---------------------------
def main():
    db_init()
    log("Forwarder started")
    log("Paperless:", PAPERLESS_BASE_URL)
    log("Ollama:", OLLAMA_BASE_URL)

    wait_for_paperless()
    wait_for_ollama()

    # create tags once
    tags = {
        "ai": get_or_create_tag_id(TAG_AI),
        "invoice": get_or_create_tag_id(TAG_INVOICE),
        "forwarded": get_or_create_tag_id(TAG_FORWARDED),
        "not_forwarded": get_or_create_tag_id(TAG_NOT_FORWARDED),
        "reason_iban": get_or_create_tag_id(TAG_REASON_IBAN),
        "reason_keyword": get_or_create_tag_id(TAG_REASON_KEYWORD),
        "topic_farm": get_or_create_tag_id(TAG_TOPIC_FARM),
        "topic_it": get_or_create_tag_id(TAG_TOPIC_IT),
        "it_deductible": get_or_create_tag_id(TAG_IT_DEDUCTIBLE),
    }

    while True:
        try:
            docs = paperless_get_docs()
            log(f"Polling... got {len(docs)} docs (latest)")

            for d in docs:
                doc_id = int(d["id"])
                if already_done(doc_id):
                    continue

                text = paperless_get_text(doc_id)
                if not text.strip():
                    log(f"Doc {doc_id}: OCR text empty -> skip for now")
                    continue

                log(f"Doc {doc_id}: content_len={len(text)}")

                # --- Gate (cheap)
                gate_prompt = f"""
Return ONLY JSON:
{{
  "is_invoice": true|false,
  "confidence": 0.0,

  "is_farming_related": true|false,
  "farming_confidence": 0.0,

  "is_it_related": true|false,
  "it_confidence": 0.0,

  "notes": "short"
}}
OCR text:
{text[:6000]}
""".strip()

                gate = ollama_generate(GATE_MODEL, gate_prompt)
                log(f"Doc {doc_id}: gate={gate}")

                run_big = bool(gate.get("is_invoice")) and float(gate.get("confidence", 0.0) or 0.0) >= GATE_INVOICE_MIN

                # --- Extract (expensive, only if gate says likely invoice)
                if run_big:
                    extract_prompt = f"""
Return ONLY JSON:
{{
  "is_invoice": true|false,
  "invoice_confidence": 0.0,

  "is_farming_related": true|false,
  "farming_confidence": 0.0,

  "is_it_related": true|false,
  "it_confidence": 0.0,

  "it_deductible_for_tax": true|false,

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
                        "invoice_confidence": float(gate.get("confidence", 0.0) or 0.0),
                        "is_farming_related": bool(gate.get("is_farming_related")),
                        "farming_confidence": float(gate.get("farming_confidence", 0.0) or 0.0),
                        "is_it_related": bool(gate.get("is_it_related")),
                        "it_confidence": float(gate.get("it_confidence", 0.0) or 0.0),
                        "it_deductible_for_tax": False,
                        "notes": gate.get("notes", "")
                    }

                log(f"Doc {doc_id}: meta={meta}")

                # Always add ai tag
                add_tags_to_document(doc_id, [tags["ai"]])

                decision = decide_and_route(meta, text)
                log(f"Doc {doc_id}: decision={decision}")

                # Tag invoice status
                if decision["invoice"]:
                    add_tags_to_document(doc_id, [tags["invoice"]])

                # If forwarding, attach PDF
                forwarded_any = False
                if decision["forward_farm"] or decision["forward_it"]:
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
                        f"invoice_confidence: {decision.get('invoice_conf')}\n"
                        f"reasons: {', '.join(decision.get('reasons', []))}\n"
                        f"ai_farm: {decision.get('ai_farm')} (conf={meta.get('farming_confidence','')})\n"
                        f"ai_it: {decision.get('ai_it')} (conf={meta.get('it_confidence','')})\n"
                        f"it_deductible_for_tax: {meta.get('it_deductible_for_tax','')}\n"
                        f"iban_match: {decision.get('iban_match')}\n"
                        f"keyword_match: {decision.get('keyword_match')}\n"
                        f"notes: {meta.get('notes','')}\n"
                    )

                    if decision["forward_farm"]:
                        send_email_with_pdf(FARM_FORWARD_TO, subject, body, filename, pdf)
                        forwarded_any = True

                    if decision["forward_it"]:
                        send_email_with_pdf(IT_FORWARD_TO, subject, body, filename, pdf)
                        forwarded_any = True

                # Tags for forwarding outcome + reasons/topics
                if forwarded_any:
                    to_add = [tags["forwarded"]]
                    if decision["iban_match"]:
                        to_add.append(tags["reason_iban"])
                    if decision["keyword_match"]:
                        to_add.append(tags["reason_keyword"])
                    if decision["ai_farm"]:
                        to_add.append(tags["topic_farm"])
                    if decision["ai_it"]:
                        to_add.append(tags["topic_it"])
                    if decision["it_deductible"]:
                        to_add.append(tags["it_deductible"])
                    add_tags_to_document(doc_id, to_add)
                    log(f"Doc {doc_id}: forwarded (farm={decision['forward_farm']}, it={decision['forward_it']})")
                else:
                    if decision["invoice"]:
                        add_tags_to_document(doc_id, [tags["not_forwarded"]])
                    log(f"Doc {doc_id}: not forwarded")

                # Mark done only after processing with non-empty OCR
                mark_done(doc_id)

        except Exception as e:
            log("ERROR:", repr(e))

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
