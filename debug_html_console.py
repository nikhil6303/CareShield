from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        client = context.new_cdp_session(page)
        client.send("Network.setCacheDisabled", {"cacheDisabled": True})

        page.on("console", lambda msg: print(f"[BROWSER CONSOLE] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: print(f"[BROWSER PAGEERROR] {err}"))

        page.goto("http://127.0.0.1:5000/", wait_until="load")
        page.wait_for_timeout(3000)

        status_text = page.locator("#api-status-text").text_content()
        print(f"Status element text content: '{status_text}'")
        browser.close()

if __name__ == "__main__":
    main()
