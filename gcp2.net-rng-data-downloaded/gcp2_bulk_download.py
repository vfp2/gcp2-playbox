#!/usr/bin/env python3
"""
GCP 2.0 Bulk Data Downloader
Browser automation via Playwright to download all device and network coherence data.

Usage:
  python3 gcp2_bulk_download.py --phase probe       # Test single download, discover form
  python3 gcp2_bulk_download.py --phase device      # Download device data (history + latest)
  python3 gcp2_bulk_download.py --phase network     # Download all network monthly data
  python3 gcp2_bulk_download.py --phase both        # Download everything
  python3 gcp2_bulk_download.py --scan              # Rebuild progress from downloaded files

Device downloads include:
  - Historical data (all past data, option=1)
  - Latest data (current month only, option=0)

Login: The script opens a browser window for you to log in manually,
       then proceeds with automated downloads.

The script reads actual groups, years, and months from the live form —
no hardcoded assumptions about data availability.

Requires: pip install playwright && playwright install chromium
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = "https://gcp2.net/data-results/data-download"
LOGIN_URL = "https://gcp2.net/login"
OUTPUT_DIR = Path("/tmp/gcp2_data")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Discovered form field names (from probe)
DEVICE_SELECT = "device_1"       # select name for device ID
NETWORK_GROUP_SELECT = "group_2" # select name for network group
NETWORK_YEAR_SELECT = "year_2"   # select name for year
NETWORK_MONTH_SELECT = "month_2" # select name for month
DEVICE_BUTTON = "#button_1"      # device download button
NETWORK_BUTTON = "#button_2"     # network download button

# Timeouts
DELAY_BETWEEN_DOWNLOADS = 3  # seconds between form submissions
PAGE_LOAD_TIMEOUT = 30000    # ms
DOWNLOAD_TIMEOUT = 120000    # ms (device downloads - large files)
NETWORK_DOWNLOAD_TIMEOUT = 15000  # ms (network downloads)
SELECT_TIMEOUT = 5000        # ms (for select_option to find value)


# ── Progress tracking ──────────────────────────────────────────────────────

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "device_history_done": [],  # Historical data per device
        "device_latest_done": [],   # Current month data per device
        "network_done": [],
        "errors": []
    }

def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def scan_existing_downloads(output_dir, known_groups=None):
    """Rebuild progress by scanning downloaded files on disk."""
    progress = {
        "device_history_done": [],
        "device_latest_done": [],
        "network_done": [],
        "errors": []
    }

    # Scan device downloads: output_dir/devices/{id}/*.zip
    # Files are named *_History.csv.zip or *_Latest.csv.zip
    devices_dir = output_dir / "devices"
    if devices_dir.exists():
        for device_dir in sorted(devices_dir.iterdir()):
            if not device_dir.is_dir():
                continue
            device_id = device_dir.name
            for zip_file in device_dir.glob("*.zip"):
                fname = zip_file.name
                if "_History" in fname:
                    progress["device_history_done"].append(f"device_{device_id}_history")
                elif "_Latest" in fname:
                    progress["device_latest_done"].append(f"device_{device_id}_latest")

    # Scan network downloads: output_dir/network/{safe_group}/{year}/*.zip
    network_dir = output_dir / "network"
    if network_dir.exists():
        for group_dir in sorted(network_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            for year_dir in sorted(group_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                for zip_file in sorted(year_dir.glob("*.zip")):
                    fname = zip_file.stem  # remove .zip (might still have .csv)
                    match = re.search(r'(\d{4})_(\d{2})', fname)
                    if match:
                        year = int(match.group(1))
                        month = int(match.group(2))
                        safe_name = group_dir.name
                        # Try to match against known groups
                        groups_to_check = known_groups if known_groups else []
                        for group in groups_to_check:
                            if group.lower().replace(" ", "_").replace("-", "_") == safe_name:
                                key = f"network_{group}_{year}_{month:02d}"
                                progress["network_done"].append(key)
                                break

    return progress


def merge_progress(file_progress, scan_progress):
    """Merge progress.json with scanned files, keeping the union."""
    # Handle legacy "device_done" key by treating it as device_history_done
    legacy_device = file_progress.get("device_done", [])
    legacy_history = [k.replace("device_", "device_") + "_history"
                      for k in legacy_device if not k.endswith("_history") and not k.endswith("_latest")]

    merged = {
        "device_history_done": list(set(
            file_progress.get("device_history_done", []) +
            scan_progress.get("device_history_done", []) +
            legacy_history
        )),
        "device_latest_done": list(set(
            file_progress.get("device_latest_done", []) +
            scan_progress.get("device_latest_done", [])
        )),
        "network_done": list(set(
            file_progress.get("network_done", []) +
            scan_progress.get("network_done", [])
        )),
        "errors": file_progress.get("errors", []),
    }
    return merged


# ── Form discovery ────────────────────────────────────────────────────────

def discover_device_ids(page):
    """Read all device IDs from the live form."""
    options = page.query_selector_all(f"select[name='{DEVICE_SELECT}'] option")
    device_ids = []
    for opt in options:
        val = opt.get_attribute("value")
        if val:  # skip placeholder
            device_ids.append(val)
    return device_ids


def discover_network_options(page):
    """Read actual groups, years, and per-year month options from the live form.

    Returns:
        groups: list of group option values (e.g. "Global Network", "Cluster London")
        year_months: dict {year: [month, ...]} with available months per year
    """
    # Read actual group values
    groups = []
    for opt in page.query_selector_all(f"select[name='{NETWORK_GROUP_SELECT}'] option"):
        val = opt.get_attribute("value")
        text = (opt.text_content() or "").strip()
        if val:
            groups.append(val)
    print(f"  Discovered {len(groups)} network groups:")
    for g in groups:
        print(f"    {g}")

    # Read actual years
    years = []
    for opt in page.query_selector_all(f"select[name='{NETWORK_YEAR_SELECT}'] option"):
        val = opt.get_attribute("value")
        if val:
            years.append(val)
    print(f"  Discovered {len(years)} years: {years}")

    # For each year, select it and read available months from dynamically populated dropdown
    year_months = {}
    for year in years:
        page.select_option(f"select[name='{NETWORK_YEAR_SELECT}']", value=year)
        time.sleep(1.5)  # Wait for populateMonths() JS

        months = []
        for opt in page.query_selector_all(f"select[name='{NETWORK_MONTH_SELECT}'] option"):
            val = opt.get_attribute("value")
            if val:
                months.append(val)
        year_months[year] = months
        print(f"    {year}: months {months}")

    return groups, year_months


# ── Login ──────────────────────────────────────────────────────────────────

def wait_for_login(page):
    """Navigate to login page and wait for user to authenticate."""
    print("Navigating to login page...")
    page.goto(LOGIN_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")

    # Check if already logged in (page might redirect to dashboard)
    if "/login" not in page.url:
        print(f"Already logged in (redirected to {page.url})")
        return True

    print("\n" + "=" * 60)
    print("MANUAL LOGIN REQUIRED")
    print("Please log in to gcp2.net in the browser window.")
    print("The script will continue automatically after login.")
    print("=" * 60 + "\n")

    # Wait for navigation away from login page (max 5 minutes)
    try:
        page.wait_for_url(lambda url: "/login" not in url, timeout=300000)
        print(f"Login successful! Redirected to: {page.url}")
        return True
    except Exception:
        print("Login timeout (5 minutes). Please restart and try again.")
        return False


# ── Phase 0: Probe ────────────────────────────────────────────────────────

def probe(page):
    """Discover form structure and test a single download."""
    print("\n=== PROBE: Discovering form and testing download ===")

    probe_dir = OUTPUT_DIR / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    page.goto(BASE_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
    print(f"Page loaded: {page.title()}")

    # Discover form structure
    print("\nForm analysis:")
    forms = page.query_selector_all("form")
    for i, form in enumerate(forms):
        form_id = form.get_attribute("id") or "(no id)"
        action = form.get_attribute("action") or "(no action attr)"
        method = form.get_attribute("method") or "(no method attr)"
        print(f"  Form {i}: id={form_id}, action={action}, method={method}")

    # Device form fields
    print("\nDevice form (#deviceCoherence) fields:")
    device_form = page.query_selector("#deviceCoherence")
    if device_form:
        for el in device_form.query_selector_all("select, input, textarea"):
            tag = el.evaluate("e => e.tagName")
            name = el.get_attribute("name") or "(unnamed)"
            el_type = el.get_attribute("type") or ""
            value = el.get_attribute("value") or ""
            print(f"  <{tag}> name='{name}' type='{el_type}' value='{value}'")

    # Discover all options
    print("\nDevice IDs:")
    device_ids = discover_device_ids(page)
    print(f"  {len(device_ids)} devices found")

    print("\nNetwork options:")
    groups, year_months = discover_network_options(page)

    # All buttons
    print("\nAll buttons:")
    buttons = page.query_selector_all("button")
    for b in buttons:
        btn_id = b.get_attribute("id") or "(no id)"
        btn_type = b.get_attribute("type") or "(no type)"
        btn_text = b.text_content().strip()
        visible = b.is_visible()
        print(f"  Button: id={btn_id}, type={btn_type}, text='{btn_text}', visible={visible}")

    # Test download device 58089
    print("\nAttempting download for device 58089 (all historical)...")
    try:
        page.select_option(f"select[name='{DEVICE_SELECT}']", value="58089")
        print("  Selected device 58089")
        time.sleep(0.5)

        page.check("input[name='device_coherence_option'][value='1']")
        print("  Selected 'All Historical' radio option")
        time.sleep(0.5)

        button = page.query_selector(DEVICE_BUTTON)
        if button:
            print(f"  Button visible: {button.is_visible()}")
            with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as download_info:
                button.click()
                print("  Clicked download button, waiting for download...")

            download = download_info.value
            dest = probe_dir / download.suggested_filename
            download.save_as(str(dest))
            print(f"  Downloaded: {dest}")
            print(f"  Size: {dest.stat().st_size} bytes")

            if dest.suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(dest) as zf:
                    print(f"  ZIP contents: {zf.namelist()}")
                    for name in zf.namelist():
                        with zf.open(name) as f:
                            lines = f.read().decode('utf-8').split('\n')
                            print(f"  {name}: {len(lines)} lines")
                            for line in lines[:5]:
                                print(f"    {line}")
            else:
                print(f"  Content preview:\n    {dest.read_text()[:200]}")

    except Exception as e:
        print(f"  ERROR: {e}")
        page.screenshot(path=str(probe_dir / "after_click.png"))
        print(f"  Screenshot saved to {probe_dir}/after_click.png")


# ── Download helper ────────────────────────────────────────────────────────

def click_and_wait_for_download(page, button_selector, timeout_seconds=60):
    """Click button, then poll for either a download event or a 'File Not Found' error page.

    Returns the Playwright download object on success.
    Raises Exception immediately if the error page appears or on timeout.
    """
    downloads = []

    def on_download(dl):
        downloads.append(dl)

    page.on("download", on_download)

    try:
        page.click(button_selector)

        # Poll every 0.5s for download or error page
        for _ in range(timeout_seconds * 2):
            if downloads:
                return downloads[0]

            # Detect GCP2 "File Not Found" error page
            try:
                body = page.inner_text("body", timeout=1000)
                if "file you requested was not found" in body.lower():
                    raise Exception("File Not Found - no data for this selection")
            except Exception as e:
                if "File Not Found" in str(e):
                    raise
                # inner_text can fail if page is navigating, ignore

            time.sleep(0.5)

        raise Exception("Download timeout - no file received")
    finally:
        page.remove_listener("download", on_download)


# ── Phase 1: Device Coherence ─────────────────────────────────────────────

def download_device_data(page, device_id, output_dir, progress, mode="history"):
    """Download device data.

    Args:
        mode: "history" for all historical data (option=1),
              "latest" for current month only (option=0)
    """
    if mode == "history":
        key = f"device_{device_id}_history"
        progress_key = "device_history_done"
        option_value = "1"
    else:
        key = f"device_{device_id}_latest"
        progress_key = "device_latest_done"
        option_value = "0"

    if key in progress[progress_key]:
        return True

    dest_dir = output_dir / "devices" / str(device_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        page.goto(BASE_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")

        # Select device
        page.select_option(f"select[name='{DEVICE_SELECT}']", value=str(device_id))
        time.sleep(0.5)

        # Select radio option (0=Latest, 1=All Historical)
        page.check(f"input[name='device_coherence_option'][value='{option_value}']")
        time.sleep(0.3)

        # Click download — detects "File Not Found" error page immediately
        download = click_and_wait_for_download(page, DEVICE_BUTTON,
                                               timeout_seconds=DOWNLOAD_TIMEOUT // 1000)
        filename = download.suggested_filename or f"device_{device_id}_{mode}.zip"
        dest = dest_dir / filename
        download.save_as(str(dest))

        progress[progress_key].append(key)
        save_progress(progress)
        return True

    except Exception as e:
        err_str = str(e)
        if "Call log:" in err_str:
            err_str = err_str.split("Call log:")[0].strip()
        progress["errors"].append({
            "key": key, "error": err_str,
            "time": datetime.now().isoformat()
        })
        save_progress(progress)
        return False


def phase_device(page, progress):
    """Download all device data (both historical and latest).

    Reads device list from the live form.
    """
    print("\nDiscovering devices from form...")
    page.goto(BASE_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
    device_ids = discover_device_ids(page)
    total = len(device_ids)

    # Count what's already done
    history_done = sum(1 for d in device_ids
                       if f"device_{d}_history" in progress["device_history_done"])
    latest_done = sum(1 for d in device_ids
                      if f"device_{d}_latest" in progress["device_latest_done"])

    print(f"\n=== DEVICE COHERENCE: {total} devices ===")
    print(f"    Historical: {history_done}/{total} done")
    print(f"    Latest:     {latest_done}/{total} done\n")

    # Download historical data for all devices
    print("--- Downloading HISTORICAL data ---")
    for i, device_id in enumerate(device_ids):
        key = f"device_{device_id}_history"
        if key in progress["device_history_done"]:
            continue

        print(f"[{i+1}/{total}] Device {device_id} (history)...", end=" ", flush=True)
        ok = download_device_data(page, device_id, OUTPUT_DIR, progress, mode="history")
        print("OK" if ok else "FAILED")
        time.sleep(DELAY_BETWEEN_DOWNLOADS)

    # Download latest (current month) data for all devices
    print("\n--- Downloading LATEST (current month) data ---")
    for i, device_id in enumerate(device_ids):
        key = f"device_{device_id}_latest"
        if key in progress["device_latest_done"]:
            continue

        print(f"[{i+1}/{total}] Device {device_id} (latest)...", end=" ", flush=True)
        ok = download_device_data(page, device_id, OUTPUT_DIR, progress, mode="latest")
        print("OK" if ok else "FAILED")
        time.sleep(DELAY_BETWEEN_DOWNLOADS)

    history_done = sum(1 for d in device_ids
                       if f"device_{d}_history" in progress["device_history_done"])
    latest_done = sum(1 for d in device_ids
                      if f"device_{d}_latest" in progress["device_latest_done"])
    print(f"\nDevice phase complete: {history_done}/{total} history, {latest_done}/{total} latest")


# ── Phase 2: Network Coherence ────────────────────────────────────────────

def download_network_data(page, group, year, month, output_dir, progress):
    """Download network coherence for one group/year/month."""
    key = f"network_{group}_{year}_{month:02d}"
    if key in progress["network_done"]:
        return True

    safe_group = group.lower().replace(" ", "_").replace("-", "_")
    dest_dir = output_dir / "network" / safe_group / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        page.goto(BASE_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")

        # Select group (use short timeout to detect invalid values fast)
        page.select_option(f"select[name='{NETWORK_GROUP_SELECT}']", value=group,
                           timeout=SELECT_TIMEOUT)

        # Select year (triggers month population via JS)
        page.select_option(f"select[name='{NETWORK_YEAR_SELECT}']", value=str(year),
                           timeout=SELECT_TIMEOUT)
        time.sleep(1)  # Wait for populateMonths() JS

        # Select month
        page.select_option(f"select[name='{NETWORK_MONTH_SELECT}']", value=str(month),
                           timeout=SELECT_TIMEOUT)
        time.sleep(0.5)

        # Click download — detects "File Not Found" error page immediately
        download = click_and_wait_for_download(page, NETWORK_BUTTON,
                                               timeout_seconds=NETWORK_DOWNLOAD_TIMEOUT // 1000)
        filename = download.suggested_filename or f"{safe_group}_{year}_{month:02d}.zip"
        dest = dest_dir / filename
        download.save_as(str(dest))

        progress["network_done"].append(key)
        save_progress(progress)
        return True

    except Exception as e:
        err_str = str(e)
        # Shorten verbose Playwright error messages for readability
        if "Call log:" in err_str:
            err_str = err_str.split("Call log:")[0].strip()
        progress["errors"].append({
            "key": key, "error": err_str,
            "time": datetime.now().isoformat()
        })
        save_progress(progress)
        return False


def phase_network(page, progress):
    """Download all network data. Reads groups/years/months from the live form."""
    print("\nDiscovering network options from form...")
    page.goto(BASE_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
    groups, year_months = discover_network_options(page)

    # Count total downloadable combinations
    total_downloads = 0
    all_combos = []
    for group in groups:
        for year, months in sorted(year_months.items()):
            for month in months:
                all_combos.append((group, int(year), int(month)))
                total_downloads += 1

    already_done = sum(
        1 for g, y, m in all_combos
        if f"network_{g}_{y}_{m:02d}" in progress.get("network_done", [])
    )

    print(f"\n=== NETWORK COHERENCE: {len(groups)} groups x "
          f"{len(year_months)} years = {total_downloads} downloads "
          f"({already_done} already done) ===\n")

    count = 0
    for group in groups:
        for year, months in sorted(year_months.items()):
            for month_str in months:
                month = int(month_str)
                year_int = int(year)
                key = f"network_{group}_{year_int}_{month:02d}"
                if key in progress["network_done"]:
                    continue

                count += 1
                print(f"[{count}] {group} {year}-{month:02d}...", end=" ", flush=True)
                ok = download_network_data(page, group, year_int, month, OUTPUT_DIR, progress)
                print("OK" if ok else "FAILED")
                time.sleep(DELAY_BETWEEN_DOWNLOADS)

    done = len(progress["network_done"])
    print(f"\nNetwork phase complete: {done} downloaded")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GCP 2.0 Bulk Data Downloader")
    parser.add_argument("--phase", choices=["probe", "device", "network", "both"],
                        default="probe", help="Download phase")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--scan", action="store_true",
                        help="Scan output directory to rebuild progress, then exit")
    args = parser.parse_args()

    output_dir = Path(args.output)
    progress_file = output_dir / "progress.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update module-level paths used by load/save_progress
    import __main__
    __main__.OUTPUT_DIR = output_dir
    __main__.PROGRESS_FILE = progress_file

    # Load existing progress from file
    file_progress = load_progress()

    # Scan downloaded files and merge with progress.json
    # (pass None for groups since we don't know them yet without the form)
    scan_progress = scan_existing_downloads(output_dir, known_groups=None)
    progress = merge_progress(file_progress, scan_progress)

    # Clear old errors for items that are now done
    done_keys = set(
        progress["device_history_done"] +
        progress["device_latest_done"] +
        progress["network_done"]
    )
    progress["errors"] = [e for e in progress["errors"] if e.get("key") not in done_keys]
    save_progress(progress)

    print(f"Progress: {len(progress['device_history_done'])} device history, "
          f"{len(progress['device_latest_done'])} device latest, "
          f"{len(progress['network_done'])} network months "
          f"({len(progress['errors'])} previous errors)")

    if args.scan:
        print("\nScan complete. Progress saved.")
        print(f"  Device historical: {len(progress['device_history_done'])}")
        print(f"  Device latest: {len(progress['device_latest_done'])}")
        print(f"  Network downloads: {len(progress['network_done'])}")
        return

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # Step 1: Login
        if not wait_for_login(page):
            browser.close()
            sys.exit(1)

        # Step 2: Run requested phase
        if args.phase == "probe":
            probe(page)
        elif args.phase == "device":
            phase_device(page, progress)
        elif args.phase == "network":
            phase_network(page, progress)
        elif args.phase == "both":
            phase_device(page, progress)
            phase_network(page, progress)

        browser.close()

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Device historical downloads: {len(progress['device_history_done'])}")
    print(f"Device latest downloads: {len(progress['device_latest_done'])}")
    print(f"Network downloads: {len(progress['network_done'])}")
    print(f"Errors: {len(progress['errors'])}")
    if progress["errors"]:
        print("Recent errors:")
        for e in progress["errors"][-10:]:
            print(f"  {e['key']}: {e['error'][:100]}")


if __name__ == "__main__":
    main()
