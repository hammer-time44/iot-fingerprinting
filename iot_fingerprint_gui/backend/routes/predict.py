"""
Prediction endpoints for timestamps and uploaded capture files.
"""
import io
import csv
import dpkt
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..pipeline import (
    process_packet_timestamps,
    CALIBRATED_THRESHOLD,
    compute_raw_stats,
    scale_features,
    predict_window
)

router = APIRouter()


class TimestampRequest(BaseModel):
    timestamps: List[float] = Field(..., description="Epoch packet timestamps in seconds")
    threshold: Optional[float] = Field(CALIBRATED_THRESHOLD, description="Confidence threshold for known vs unknown")


@router.post("/predict/timestamps")
def predict_from_timestamps(req: TimestampRequest):
    """Predict device label and known/unknown status from a list of packet timestamps."""
    if len(req.timestamps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 timestamps are required.")

    thresh = req.threshold if req.threshold is not None else CALIBRATED_THRESHOLD
    windows, summary = process_packet_timestamps(req.timestamps, threshold=thresh)

    if "error" in summary:
        raise HTTPException(status_code=400, detail=summary["error"])

    return {
        "windows": windows,
        "summary": summary
    }


@router.post("/predict/upload")
async def predict_from_file(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(CALIBRATED_THRESHOLD),
    device_name: Optional[str] = Form("Custom Device")
):
    """
    Extract timestamps from an uploaded .pcap, .pcapng, or .csv file and run the fingerprinting pipeline.
    Payloads are never read or stored; only packet arrival timestamps are extracted.
    """
    filename = file.filename or "uploaded_file"
    ext = filename.lower().split(".")[-1]

    contents = await file.read()
    timestamps: List[float] = []

    try:
        if ext in ["pcap", "pcapng", "cap"]:
            f_io = io.BytesIO(contents)
            try:
                reader = dpkt.pcapng.Reader(f_io)
                for ts, _ in reader:
                    timestamps.append(float(ts))
            except Exception:
                f_io.seek(0)
                try:
                    reader = dpkt.pcap.Reader(f_io)
                    for ts, _ in reader:
                        timestamps.append(float(ts))
                except Exception as pcap_err:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to parse PCAP file: {str(pcap_err)}"
                    )
        elif ext in ["csv", "txt"]:
            text = contents.decode("utf-8", errors="ignore")
            reader = csv.reader(text.splitlines())
            rows = list(reader)
            if not rows:
                raise HTTPException(status_code=400, detail="Empty CSV file.")

            # Identify timestamp column
            header = [h.strip().lower() for h in rows[0]]
            ts_idx = -1
            for col_candidate in ["timestamp", "ts", "time", "epoch"]:
                if col_candidate in header:
                    ts_idx = header.index(col_candidate)
                    break

            start_row = 1 if ts_idx != -1 else 0
            if ts_idx == -1:
                ts_idx = 0  # default to first column

            for r in rows[start_row:]:
                if len(r) > ts_idx:
                    try:
                        val = float(r[ts_idx].strip())
                        timestamps.append(val)
                    except ValueError:
                        continue
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '.{ext}'. Supported formats are .pcap, .pcapng, .csv, and .txt."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    if len(timestamps) < 129:
        raise HTTPException(
            status_code=400,
            detail=f"File contains {len(timestamps)} packets. At least 129 packets are required to form a full 128-packet IAT window."
        )

    thresh = threshold if threshold is not None else CALIBRATED_THRESHOLD
    windows, summary = process_packet_timestamps(timestamps, threshold=thresh)
    summary["entered_device_name"] = device_name
    summary["filename"] = filename

    return {
        "filename": filename,
        "packet_count": len(timestamps),
        "windows": windows,
        "summary": summary
    }
