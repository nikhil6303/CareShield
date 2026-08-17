from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "patient_churn_dataset.csv.xls"
SCREENSHOT_PATH = ROOT / "html-upload-success.png"

def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Navigate to HTML frontend on Flask server
        page.goto("http://127.0.0.1:5000/", wait_until="networkidle")
        expect(page.get_by_text("Upload Member Dataset")).to_be_visible(timeout=15000)
        expect(page.get_by_text("REST API: Online")).to_be_visible(timeout=10000)

        # Upload dataset
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(str(DATASET_PATH))
        expect(page.get_by_text(DATASET_PATH.name)).to_be_visible(timeout=10000)

        # Click analyze dataset
        page.get_by_role("button", name="Analyze Dataset").click()
        expect(page.get_by_text("Strategic Retention Dashboard")).to_be_visible(timeout=120000)
        expect(page.get_by_text("2,000")).to_be_visible(timeout=10000)

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()

        print("HTML upload UI test PASSED successfully!")
        print(f"Screenshot saved to: {SCREENSHOT_PATH}")

if __name__ == "__main__":
    main()
