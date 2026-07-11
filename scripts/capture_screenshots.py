import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:8512"


def wait_for_server(timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1).close()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Streamlit did not start within 30 seconds")


def capture(page, tab_name: str, filename: str, scroll_to: str | None = None) -> None:
    page.get_by_role("tab", name=tab_name, exact=True).click()
    page.wait_for_function("document.querySelectorAll('[data-testid=stSkeleton]').length === 0", timeout=30000)
    page.wait_for_timeout(5000)
    page.evaluate("window.scrollTo(0, 0)")
    if scroll_to:
        page.get_by_text(scroll_to, exact=True).scroll_into_view_if_needed()
        page.mouse.wheel(0, 420)
        page.wait_for_timeout(400)
    page.screenshot(path=OUTPUT / filename, full_page=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "streamlit"),
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=8512",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_server()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(8000)
            capture(page, "Overview", "overview-dashboard.png")
            capture(page, "Add Invoices", "add-invoices-queue.png")
            capture(page, "Approvals", "invoice-detail-risk.png", scroll_to="Open invoice detail")
            capture(page, "Risk Rules", "risk-rules.png")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
