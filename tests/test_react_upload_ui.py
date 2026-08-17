from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "patient_churn_dataset.csv.xls"
SCREENSHOT_PATH = ROOT / "react-upload-success.png"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto("http://localhost:3000/", wait_until="networkidle")
        expect(page.get_by_text("Upload Member Dataset")).to_be_visible(timeout=15000)

        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(str(DATASET_PATH))
        expect(page.get_by_text(DATASET_PATH.name)).to_be_visible(timeout=10000)

        with page.expect_response(
            lambda response: response.url.endswith("/predict-file") and response.request.method == "POST",
            timeout=120000,
        ) as upload_response:
            page.get_by_role("button", name="Analyze Dataset").click()

        response = upload_response.value
        assert response.status == 200, response.text()
        payload = response.json()
        assert payload["success"] is True
        assert payload["total_members"] == 2000
        assert payload["members"][0]["drivers"]
        assert payload["members"][0]["recommendations"]

        expect(page.get_by_text("Strategic Retention Dashboard")).to_be_visible(timeout=10000)
        expect(page.get_by_text("Risk Distribution").first).to_be_visible(timeout=10000)
        expect(page.get_by_text("High-Risk Members Priority Action List")).to_be_visible(timeout=10000)

        # Click member C20010 row to navigate directly to Retention Advisor
        page.get_by_text("C20010").first.click()
        page.wait_for_timeout(1000)

        # Verify Retention Advisor loaded for C20010 with exact prediction and consolidated actions
        expect(page.get_by_text("Individual Member Retention Assessment")).to_be_visible(timeout=10000)
        expect(page.get_by_text("Prioritized Retention Actions")).to_be_visible(timeout=10000)
        expect(page.get_by_role("heading", name="Service Recovery")).to_be_visible(timeout=10000)

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()

        print("React upload UI test passed.")
        print(f"Upload response members: {payload['total_members']}")
        print(f"Risk summary: {payload['risk_summary']}")
        print(f"Screenshot: {SCREENSHOT_PATH}")


if __name__ == "__main__":
    main()
