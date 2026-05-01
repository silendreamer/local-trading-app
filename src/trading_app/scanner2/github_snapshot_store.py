from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
GITHUB_API_URL = "https://api.github.com"


@dataclass(frozen=True)
class GitHubSnapshotConfig:
    token: str
    repo: str
    branch: str = "snapshot-data"
    snapshot_dir: str = "snapshots"


def load_github_snapshot_config() -> GitHubSnapshotConfig | None:
    token = os.getenv("GITHUB_SNAPSHOT_TOKEN", "").strip()
    repo = os.getenv("GITHUB_SNAPSHOT_REPO", "").strip() or os.getenv("GITHUB_REPOSITORY", "").strip()
    if not token or not repo:
        return None
    return GitHubSnapshotConfig(
        token=token,
        repo=repo,
        branch=os.getenv("GITHUB_SNAPSHOT_BRANCH", "snapshot-data").strip() or "snapshot-data",
        snapshot_dir=os.getenv("GITHUB_SNAPSHOT_DIR", "snapshots").strip().strip("/") or "snapshots",
    )


class GitHubSnapshotStore:
    def __init__(self, config: GitHubSnapshotConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {config.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def save_text(self, relative_path: str, content: str, message: str) -> None:
        self.ensure_branch()
        path = self._path(relative_path)
        existing_sha = self.file_sha(path)
        payload: dict[str, Any] = {
            "message": message,
            "content": b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self.config.branch,
        }
        if existing_sha:
            payload["sha"] = existing_sha
        response = self.session.put(self.contents_url(path), json=payload, timeout=30)
        response.raise_for_status()

    def read_text(self, relative_path: str) -> str:
        path = self._path(relative_path)
        response = self.session.get(
            self.contents_url(path),
            params={"ref": self.config.branch},
            timeout=30,
        )
        if response.status_code == 404:
            raise FileNotFoundError(relative_path)
        response.raise_for_status()
        payload = response.json()
        return b64decode(payload["content"]).decode("utf-8")

    def exists(self, relative_path: str) -> bool:
        return self.file_sha(self._path(relative_path)) is not None

    def list_snapshot_names(self) -> list[str]:
        response = self.session.get(
            self.contents_url(self.config.snapshot_dir),
            params={"ref": self.config.branch},
            timeout=30,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return [
            item["name"]
            for item in payload
            if item.get("type") == "file" and str(item.get("name", "")).startswith("snapshot_")
        ]

    def delete(self, relative_path: str, message: str) -> bool:
        path = self._path(relative_path)
        sha = self.file_sha(path)
        if not sha:
            return False
        response = self.session.delete(
            self.contents_url(path),
            json={"message": message, "sha": sha, "branch": self.config.branch},
            timeout=30,
        )
        response.raise_for_status()
        return True

    def ensure_branch(self) -> None:
        ref_response = self.session.get(self.ref_url(self.config.branch), timeout=30)
        if ref_response.status_code != 404:
            ref_response.raise_for_status()
            return

        repo_response = self.session.get(f"{GITHUB_API_URL}/repos/{self.config.repo}", timeout=30)
        repo_response.raise_for_status()
        default_branch = repo_response.json()["default_branch"]
        default_ref = self.session.get(self.ref_url(default_branch), timeout=30)
        default_ref.raise_for_status()
        sha = default_ref.json()["object"]["sha"]
        create_response = self.session.post(
            f"{GITHUB_API_URL}/repos/{self.config.repo}/git/refs",
            json={"ref": f"refs/heads/{self.config.branch}", "sha": sha},
            timeout=30,
        )
        if create_response.status_code not in {201, 422}:
            create_response.raise_for_status()

    def file_sha(self, path: str) -> str | None:
        response = self.session.get(
            self.contents_url(path),
            params={"ref": self.config.branch},
            timeout=30,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("sha")

    def contents_url(self, path: str) -> str:
        return f"{GITHUB_API_URL}/repos/{self.config.repo}/contents/{path}"

    def ref_url(self, branch: str) -> str:
        return f"{GITHUB_API_URL}/repos/{self.config.repo}/git/ref/heads/{branch}"

    def _path(self, relative_path: str) -> str:
        return str(Path(self.config.snapshot_dir) / relative_path).replace("\\", "/")
