"""Study-claim provenance must reflect the source's real origin (GitHub #52).

The hole: claims were written with a blanket external_source (+0.10 trusted boost),
so web-scraped data could launder untrusted content into trusted self-knowledge,
bypassing the web_scrap firewall. Fixed: derive provenance from the source bucket.
"""
from __future__ import annotations

import pytest

try:
    from library.study import _provenance_for_study_claim
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("library.study import unavailable", allow_module_level=True)


@pytest.mark.parametrize("src,expected", [
    ({"source_type": "url", "url": "http://example.com/page"}, "web_scrap"),   # the fix
    ({"source_type": "web", "url": "http://x.io"}, "web_scrap"),
    ({"source_type": "doi", "doi": "10.1/abc"}, "external_source"),            # validated stays trusted
    ({"source_type": "peer_reviewed"}, "external_source"),
    ({"source_type": "codebase"}, "observed"),
    ({"source_type": "memory"}, "observed"),
    ({"source_type": "introspection"}, "observed"),
    ({"source_type": "local_file"}, "user_claim"),
    ({"source_type": "user_note"}, "user_claim"),
    ({"source_type": "something_weird"}, "web_scrap"),                         # unknown → conservative
    ({}, "web_scrap"),                                                         # empty → conservative
])
def test_provenance_reflects_source_origin(src, expected):
    assert _provenance_for_study_claim(src) == expected


def test_web_never_gets_trusted_boost():
    # The core invariant: a web/scraped source must never produce external_source.
    for st in ("url", "web"):
        assert _provenance_for_study_claim({"source_type": st, "url": "http://q"}) != "external_source"
