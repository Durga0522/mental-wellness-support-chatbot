from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4
from datetime import datetime
import uvicorn

app = FastAPI(title="Distributed Log Monitoring System")

logs_db = []
alerts_db = []


class LogEntry(BaseModel):
    timestamp: datetime
    server_id: str
    service_name: str
    log_level: str = Field(..., pattern="^(INFO|WARN|ERROR|CRITICAL)$")
    message: str
    host: str
    environment: str
    source: str


@app.get("/")
def home():
    return {"message": "Log Monitoring System Running Successfully"}


@app.post("/api/logs")
def ingest_log(log: LogEntry):
    log_data = {
        "id": str(uuid4()),
        "timestamp": log.timestamp.isoformat(),
        "server_id": log.server_id,
        "service_name": log.service_name,
        "log_level": log.log_level,
        "message": log.message,
        "host": log.host,
        "environment": log.environment,
        "source": log.source,
    }

    logs_db.append(log_data)

    if log.log_level in ["ERROR", "CRITICAL"]:
        alert_data = {
            "id": str(uuid4()),
            "alert_id": str(uuid4()),
            "timestamp": log.timestamp.isoformat(),
            "service_name": log.service_name,
            "log_level": log.log_level,
            "message": log.message,
            "status": "OPEN",
        }
        alerts_db.append(alert_data)

    return {"message": "Log stored successfully", "log": log_data}


@app.get("/api/logs")
def get_logs(
    level: Optional[str] = None,
    service_name: Optional[str] = None,
    keyword: Optional[str] = None,
    environment: Optional[str] = None,
    page: int = 1,
    size: int = 10,
):
    filtered_logs = logs_db.copy()

    if level:
        filtered_logs = [log for log in filtered_logs if log["log_level"] == level]

    if service_name:
        filtered_logs = [log for log in filtered_logs if log["service_name"] == service_name]

    if environment:
        filtered_logs = [log for log in filtered_logs if log["environment"] == environment]

    if keyword:
        filtered_logs = [
            log for log in filtered_logs
            if keyword.lower() in log["message"].lower()
        ]

    filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)

    start = (page - 1) * size
    end = start + size

    return {
        "total": len(filtered_logs),
        "page": page,
        "size": size,
        "logs": filtered_logs[start:end]
    }


@app.get("/api/alerts")
def get_alerts():
    sorted_alerts = sorted(alerts_db, key=lambda x: x["timestamp"], reverse=True)
    return {"alerts": sorted_alerts}


@app.put("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str):
    for alert in alerts_db:
        if alert["alert_id"] == alert_id:
            alert["status"] = "RESOLVED"
            return {"message": "Alert resolved", "alert": alert}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/api/dashboard/summary")
def summary():
    total_logs = len(logs_db)
    info_count = len([log for log in logs_db if log["log_level"] == "INFO"])
    warn_count = len([log for log in logs_db if log["log_level"] == "WARN"])
    error_count = len([log for log in logs_db if log["log_level"] == "ERROR"])
    critical_count = len([log for log in logs_db if log["log_level"] == "CRITICAL"])
    open_alerts = len([alert for alert in alerts_db if alert["status"] == "OPEN"])

    return {
        "total_logs": total_logs,
        "info": info_count,
        "warn": warn_count,
        "error": error_count,
        "critical": critical_count,
        "open_alerts": open_alerts
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)