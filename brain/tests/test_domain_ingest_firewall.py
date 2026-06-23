"""Red-team pins for the capability-domain ingest path/secret firewall (Matrix-v2 hardening).

The v2 local-file reader must never exfiltrate secrets even if a future "learn this folder" on-ramp
points it at a sensitive path. These pins lock the always-on protection + the optional allowlist root.
"""
from cognition.capability_domains import ingest as ing


def test_blocks_secret_named_files():
    for p in ("/home/u/.env", "/home/u/project/.env", "/home/u/secrets.txt",
              "/home/u/id_rsa", "/home/u/key.pem", "/home/u/aws_credentials", "/home/u/.netrc"):
        ok, _reason = ing._ingest_path_allowed(p)
        assert not ok, p


def test_blocks_sensitive_dirs():
    for p in ("/home/u/.ssh/config", "/home/u/.jarvis/identity.json",
              "/home/u/.git/config", "/home/u/proj/.venv/x.txt", "/home/u/.aws/creds"):
        ok, reason = ing._ingest_path_allowed(p)
        assert not ok and reason in ("sensitive_dir", "secret_name"), (p, reason)


def test_allows_ordinary_doc(monkeypatch):
    monkeypatch.delenv("JARVIS_DOMAIN_INGEST_ROOT", raising=False)
    ok, reason = ing._ingest_path_allowed("/home/u/docs/thermodynamics.md")
    assert ok, reason


def test_content_secret_scrub():
    assert ing._content_has_secret("-----BEGIN OPENSSH PRIVATE KEY-----\nx")
    assert ing._content_has_secret("aws_secret_access_key = abc123")
    assert ing._content_has_secret("token ghp_" + "a" * 36)
    assert not ing._content_has_secret("The boiling point of water is 100C.")


def test_allowlist_root_blocks_escape(monkeypatch, tmp_path):
    root = tmp_path / "ingest_root"
    root.mkdir()
    monkeypatch.setenv("JARVIS_DOMAIN_INGEST_ROOT", str(root))
    inside = root / "a.md"
    inside.write_text("x")
    ok, _ = ing._ingest_path_allowed(str(inside))
    assert ok                                   # inside the allowlisted root -> allowed
    ok, reason = ing._ingest_path_allowed(str(root / ".." / "outside.md"))
    assert not ok and reason == "outside_allowlist_root", reason   # parent-traversal escape blocked
