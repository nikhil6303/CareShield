import io
import os
import subprocess
import time

import pandas as pd
import requests
import xlwt


API_URL = "http://127.0.0.1:5000"
DATASET_PATH = "patient_churn_dataset.csv.xls"
SCRATCH_FILES = [
    "scratch_upload.csv",
    "scratch_upload.xlsx",
    "scratch_upload.xls",
    "scratch_bad_columns.csv",
    "scratch_bad_types.csv",
]


def wait_for_api(timeout_seconds=60):
    for second in range(timeout_seconds):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return second
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("Flask API did not become ready in time.")


def start_api_if_needed():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("    Existing Flask API is already running.")
            return None
    except requests.RequestException:
        pass

    print("    Starting Flask API in the background...")
    return subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def post_file(path, upload_name=None, content_type="application/octet-stream"):
    with open(path, "rb") as handle:
        files = {"file": (upload_name or os.path.basename(path), handle, content_type)}
        return requests.post(f"{API_URL}/predict-file", files=files, timeout=120)


def assert_successful_upload(response, label):
    print(f"    {label} status: {response.status_code}")
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["success"] is True
    assert data["status"] == "success"
    assert data["total_members"] > 0
    assert len(data["members"]) == data["total_members"]
    assert set(data["summary"]["risk_distribution"]).issuperset({"Low", "Medium", "High", "Critical"})

    first_member = data["members"][0]
    assert "member_id" in first_member
    assert "churn_probability" in first_member
    assert "risk_level" in first_member
    assert "drivers" in first_member
    assert "recommendations" in first_member

    print(f"    {label} members analyzed: {data['total_members']}")
    print(f"    {label} risk distribution: {data['summary']['risk_distribution']}")
    print(f"    {label} first member: {first_member['member_id']} -> {first_member['risk_level']}")
    return data


def assert_error(response, expected_fragment, label):
    print(f"    {label} status: {response.status_code}")
    body = response.json()
    print(f"    {label} response: {body}")
    assert response.status_code == 400
    assert body["success"] is False
    assert expected_fragment.lower() in body["error"].lower()


def write_legacy_xls(df, path):
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Members")

    for col_idx, column in enumerate(df.columns):
        sheet.write(0, col_idx, column)

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for col_idx, value in enumerate(row.tolist()):
            if pd.isna(value):
                sheet.write(row_idx, col_idx, "")
            else:
                if hasattr(value, "item"):
                    value = value.item()
                sheet.write(row_idx, col_idx, value)

    workbook.save(path)


def cleanup():
    for path in SCRATCH_FILES:
        if os.path.exists(path):
            os.remove(path)


print("==================================================================")
print("        MEMBER CHURN ADVISOR - REACT/FLASK UPLOAD TESTING")
print("==================================================================")

cleanup()
server_process = start_api_if_needed()

try:
    ready_after = wait_for_api()
    print(f"    Flask API ready after {ready_after} second(s).")

    print("\n[1] Health check")
    health = requests.get(f"{API_URL}/health", timeout=5)
    print(f"    GET /health: {health.status_code} {health.json()}")
    assert health.status_code == 200

    print("\n[2] CORS preflight from React dev origin")
    preflight = requests.options(
        f"{API_URL}/predict-file",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )
    print(f"    OPTIONS /predict-file: {preflight.status_code}")
    assert preflight.status_code in (200, 204)
    assert preflight.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    source_df = pd.read_csv(DATASET_PATH)
    source_df.to_csv("scratch_upload.csv", index=False)
    source_df.head(50).to_excel("scratch_upload.xlsx", index=False, engine="openpyxl")
    write_legacy_xls(source_df.head(50), "scratch_upload.xls")

    print("\n[3] CSV upload through multipart/form-data")
    csv_data = assert_successful_upload(
        post_file("scratch_upload.csv", "member_upload.csv", "text/csv"),
        "CSV upload",
    )

    print("\n[4] XLSX upload through multipart/form-data")
    assert_successful_upload(
        post_file(
            "scratch_upload.xlsx",
            "member_upload.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "XLSX upload",
    )

    print("\n[5] XLS upload through multipart/form-data")
    assert_successful_upload(
        post_file("scratch_upload.xls", "member_upload.xls", "application/vnd.ms-excel"),
        "XLS upload",
    )

    print("\n[6] Member detail cache after upload")
    sample_member_id = csv_data["members"][0]["member_id"]
    detail = requests.get(f"{API_URL}/member/{sample_member_id}", timeout=20)
    print(f"    GET /member/{sample_member_id}: {detail.status_code}")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["drivers"]
    assert detail_data["recommendations"]
    print(f"    SHAP top driver: {detail_data['drivers'][0]['label']}")
    print(f"    Top recommendation: {detail_data['recommendations'][0]['action']}")

    print("\n[7] Missing file error")
    assert_error(
        requests.post(f"{API_URL}/predict-file", timeout=10),
        "No file was uploaded",
        "Missing file",
    )

    print("\n[8] Unsupported file extension error")
    unsupported = requests.post(
        f"{API_URL}/predict-file",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        timeout=10,
    )
    assert_error(unsupported, "Unsupported file type", "Unsupported file")

    print("\n[9] Corrupted Excel file error")
    corrupted = requests.post(
        f"{API_URL}/predict-file",
        files={"file": ("broken.xlsx", io.BytesIO(b"not a workbook"), "application/octet-stream")},
        timeout=10,
    )
    assert_error(corrupted, "Could not parse .xlsx file", "Corrupted XLSX")

    print("\n[10] Missing required column error")
    bad_columns = pd.DataFrame({
        "PatientID": ["BAD001", "BAD002"],
        "Age": [40, 50],
        "Gender": ["Male", "Female"],
    })
    bad_columns.to_csv("scratch_bad_columns.csv", index=False)
    assert_error(
        post_file("scratch_bad_columns.csv", "scratch_bad_columns.csv", "text/csv"),
        "Missing required column",
        "Missing columns",
    )

    print("\n[11] Incorrect data type error")
    bad_types = source_df.copy().astype({"Age": "object"})
    bad_types.loc[0, "Age"] = "INVALID_AGE_STRING"
    bad_types.to_csv("scratch_bad_types.csv", index=False)
    assert_error(
        post_file("scratch_bad_types.csv", "scratch_bad_types.csv", "text/csv"),
        "non-numeric",
        "Bad numeric type",
    )

    print("\n==================================================================")
    print("             ALL BACKEND UPLOAD WORKFLOW TESTS PASSED")
    print("==================================================================")

finally:
    cleanup()
    if server_process is not None:
        print("\nCleaning up: terminating Flask server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
