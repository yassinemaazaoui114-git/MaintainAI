"""Capture README screenshots of the running app with Playwright.

    pip install -r requirements-dev.txt && python -m playwright install chromium
    python scripts/take_screenshots.py

Starts its own Streamlit server on a temporary copy of the database (so nothing is
written to the real one), visits each page, and saves PNGs to docs/screenshots/.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 8599
URL = f"http://localhost:{PORT}"
WIDTH, HEIGHT = 1440, 900


def wait_for_server(timeout: float = 90) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).status == 200:
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Streamlit server did not start")


def settle(page: Page) -> None:
    page.wait_for_selector("iframe", timeout=60_000)
    page.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]')", timeout=60_000)
    page.wait_for_timeout(2500)  # web fonts + iframe self-sizing


def main_height(page: Page) -> int:
    return page.evaluate("document.querySelector('[data-testid=stMain]').scrollHeight")


def go(page: Page, screen: str) -> None:
    # nav labels carry a count badge ("Machines 50"), so match the start of the label
    page.locator("[data-testid=stSidebar] [data-testid=stRadioOption]").filter(
        has_text=re.compile(rf"^{re.escape(screen)}\s*\d*$")).click()
    settle(page)


def shoot(page: Page, name: str, full: bool = False, max_height: int = 2400) -> None:
    if full:
        page.set_viewport_size({"width": WIDTH, "height": min(main_height(page), max_height)})
        page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT / f"{name}.png"))
    page.set_viewport_size({"width": WIDTH, "height": HEIGHT})
    print(f"  {name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "factory_maintenance.db", tmp / "factory_maintenance.db")
    env = {**os.environ, "MAINTAINAI_DB": str(tmp / "factory_maintenance.db")}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.goto(URL)
            settle(page)
            shoot(page, "dashboard")
            # health table with plain-words drivers, then the per-machine explanation panel
            page.evaluate("""() => {
                const main = document.querySelector('[data-testid=stMain]');
                const frames = [...document.querySelectorAll('iframe')];
                const table = frames[frames.length - 3];
                main.scrollTo(0, table.getBoundingClientRect().top + main.scrollTop - 20);
            }""")
            page.wait_for_timeout(800)
            shoot(page, "dashboard_health_table")
            page.evaluate("""() => {
                const main = document.querySelector('[data-testid=stMain]');
                main.scrollTo(0, main.scrollHeight);
            }""")
            page.wait_for_timeout(800)
            shoot(page, "dashboard_explanation")
            for screen, name in (("Machines", "machines"), ("Failure Log", "failure_log"),
                                 ("Benchmarks", "benchmarks")):
                go(page, screen)
                shoot(page, name, full=screen == "Benchmarks", max_height=3200)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
