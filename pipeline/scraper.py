"""GitHub API scraper — fetches Python source files without cloning repos."""
import json
import time
from pathlib import Path

import requests

QUERIES = [
    "language:python topic:algorithms stars:>50",
    "language:python topic:data-structures stars:>50",
    "language:python topic:statistics stars:>50",
    "language:python topic:numerical stars:>50",
    "language:python topic:scientific-computing stars:>20",
    "language:python topic:math stars:>30",
]

_EXCLUDED_PATH_PARTS = {
    "tests", "test", "__pycache__", ".venv", "venv",
    "migrations", "node_modules", "docs", "examples", "notebooks",
}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _check_rate_limit(response: requests.Response) -> None:
    remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
    if remaining < 10:
        reset = int(response.headers.get("X-RateLimit-Reset", 0))
        sleep_secs = max(0, reset - int(time.time())) + 2
        print(f"[scraper] Rate limit low ({remaining} remaining), sleeping {sleep_secs}s...")
        time.sleep(sleep_secs)


def search_repos(token: str, query: str, per_page: int = 10) -> list[dict]:
    """Search GitHub repositories. Returns list of {full_name, default_branch} dicts."""
    url = "https://api.github.com/search/repositories"
    repos = []
    for page in range(1, 6):  # max 5 pages
        resp = requests.get(
            url,
            headers=_headers(token),
            params={"q": query, "per_page": per_page, "page": page, "sort": "stars"},
            timeout=30,
        )
        _check_rate_limit(resp)
        if resp.status_code in (403, 429):
            wait = int(resp.headers.get("Retry-After", 60))
            print(f"[scraper] Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            break
        items = resp.json().get("items", [])
        if not items:
            break
        for item in items:
            repos.append({
                "full_name": item["full_name"],
                "default_branch": item.get("default_branch", "main"),
            })
        if len(items) < per_page:
            break
        time.sleep(1.0)
    return repos


def get_repo_python_files(
    token: str,
    owner: str,
    repo: str,
    branch: str = "main",
    min_size: int = 1024,
    max_size: int = 51200,
) -> list[dict]:
    """Get Python files via git/trees API. Returns list of {path, size} dicts."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=_headers(token), timeout=30)
    _check_rate_limit(resp)
    if resp.status_code == 409:  # empty repo
        return []
    if resp.status_code != 200:
        return []
    data = resp.json()
    if data.get("truncated"):
        print(f"[scraper] Tree truncated for {owner}/{repo}, using partial results")

    files = []
    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path.endswith(".py"):
            continue
        size = item.get("size", 0)
        if size < min_size or size > max_size:
            continue
        # Exclude test/cache directories
        path_parts = set(path.split("/")[:-1])
        if path_parts & _EXCLUDED_PATH_PARTS:
            continue
        # Exclude test files by name
        filename = path.split("/")[-1]
        if filename.startswith("test_") or filename.endswith("_test.py"):
            continue
        files.append({"path": path, "size": size})

    time.sleep(0.5)
    return files


def fetch_raw_content(token: str, owner: str, repo: str, branch: str, path: str) -> str | None:
    """Fetch raw file content from raw.githubusercontent.com."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(url, headers=_headers(token), timeout=30)
    _check_rate_limit(resp)
    if resp.status_code != 200:
        return None
    time.sleep(0.2)
    return resp.text


def _safe_filename(owner: str, repo: str, path: str) -> str:
    safe_path = path.replace("/", "__").replace("\\", "__")
    return f"{owner}__{repo}__{safe_path}.json"


def scrape(token: str, output_dir: Path, target_files: int = 70) -> list[Path]:
    """
    Main entry point. Fetch Python source files from GitHub and save as JSON.
    Returns list of saved file paths. Idempotent: skips already-saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    seen_repos: set[str] = set()

    # Count existing files
    existing = list(output_dir.glob("*.json"))
    files_fetched = len(existing)
    print(f"[scraper] Found {files_fetched} existing files, targeting {target_files} total")

    for query in QUERIES:
        if files_fetched >= target_files:
            break
        print(f"[scraper] Searching: {query!r}")
        try:
            repos = search_repos(token, query, per_page=10)
        except Exception as e:
            print(f"[scraper] Search failed: {e}")
            continue

        for repo_info in repos:
            if files_fetched >= target_files:
                break
            full_name = repo_info["full_name"]
            if full_name in seen_repos:
                continue
            seen_repos.add(full_name)

            owner, repo = full_name.split("/", 1)
            branch = repo_info["default_branch"]
            print(f"[scraper] Scanning: {full_name} (branch: {branch})")

            try:
                py_files = get_repo_python_files(token, owner, repo, branch)
            except Exception as e:
                print(f"[scraper] Tree fetch failed for {full_name}: {e}")
                continue

            for file_info in py_files:
                if files_fetched >= target_files:
                    break
                path = file_info["path"]
                out_path = output_dir / _safe_filename(owner, repo, path)

                if out_path.exists():
                    saved_paths.append(out_path)
                    continue

                try:
                    content = fetch_raw_content(token, owner, repo, branch, path)
                except Exception as e:
                    print(f"[scraper] Fetch failed {path}: {e}")
                    continue

                if content is None:
                    continue

                record = {"repo": full_name, "path": path, "content": content}
                out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                saved_paths.append(out_path)
                files_fetched += 1
                print(f"[scraper] Saved ({files_fetched}/{target_files}): {out_path.name}")

    print(f"[scraper] Done. Total files: {files_fetched}")
    return saved_paths
