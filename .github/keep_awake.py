"""Open the hosted demo in a real browser so Streamlit Community Cloud doesn't put it to sleep.

A plain HTTP request isn't enough: Streamlit only counts a browser session that connects to the app.
If the app is already asleep, click the wake button and wait for it to start.
Exits non-zero if the app never shows up, so the workflow run fails and GitHub sends an email.
Never logs in: it stops at the password screen, so no password is stored here.
"""
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = "https://ashishshiwlani-portfolio-qa-app-9o9bqn.streamlit.app"
READY = re.compile(r"Portfolio Q&A")


def app_text(page):
    """The app runs inside an iframe at /~/+/; return its visible text, or '' if not there yet."""
    for frame in page.frames:
        if "/~/+/" in frame.url:
            try:
                return frame.locator("body").inner_text(timeout=2000)
            except Exception:
                return ""
    return ""


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)

    wake = page.get_by_role("button", name=re.compile(r"get this app back up", re.I))
    if wake.count():
        print("App was asleep. Clicking the wake button.")
        wake.first.click()
    else:
        print("No sleep screen.")

    deadline = time.time() + 300  # waking can take a few minutes
    while time.time() < deadline:
        if READY.search(app_text(page)):
            break
        page.wait_for_timeout(5_000)
    else:
        print("App did not come up within 5 minutes.")
        page.screenshot(path="not-ready.png", full_page=True)
        browser.close()
        sys.exit(1)

    print("App is up:", app_text(page).split("\n")[0:3])
    page.wait_for_timeout(30_000)  # stay connected for a while so the visit counts
    browser.close()
