from src.backend import main, db_backend as db
from src.backend.pipeline_runner import _write_reports

with db.get_connection(main.DB_PATH) as conn:
    logs = db.list_logs(conn)

for log in logs:
    if log["status"] != "done":
        continue
    out_dir = main.OUTPUT_DIR / log["id"]
    _write_reports(main.DB_PATH, log["id"], out_dir)
    print(f"Wrote reports for {log['id']} ({log['filename']})")
