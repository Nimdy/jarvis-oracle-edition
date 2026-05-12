"""Tests for content validation, paywall detection, and Content-Type routing."""

import html
import subprocess
import unittest
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestValidateContentQuality(unittest.TestCase):

    def setUp(self):
        from autonomy.knowledge_integrator import _validate_content_quality
        self.validate = _validate_content_quality

    def test_empty_text_rejected(self):
        ok, reason = self.validate("")
        self.assertFalse(ok)
        self.assertEqual(reason, "too_short")

    def test_short_text_rejected(self):
        ok, reason = self.validate("hello world")
        self.assertFalse(ok)
        self.assertEqual(reason, "too_short")

    def test_valid_academic_text_passes(self):
        text = (
            "We propose a novel framework for reinforcement learning that combines "
            "model-based planning with model-free control. Our approach achieves "
            "state-of-the-art results on the Atari benchmark suite, outperforming "
            "prior methods by a significant margin. The key insight is that planning "
            "in a learned world model provides better sample efficiency while "
            "retaining the asymptotic performance of model-free methods. "
        ) * 3
        ok, reason = self.validate(text)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_binary_garbage_rejected(self):
        binary_chars = "\x00\x01\x02\x03\x04\x05" * 400
        ok, reason = self.validate(binary_chars)
        self.assertFalse(ok)
        self.assertIn("binary_garbage", reason)

    def test_replacement_chars_rejected(self):
        text = "a" * 100 + "\ufffd" * 50 + "b" * 100
        ok, reason = self.validate(text)
        self.assertFalse(ok)
        self.assertIn("encoding_garbage", reason)

    def test_few_replacement_chars_ok(self):
        text = "This is a normal text with a few issues " * 10 + "\ufffd" * 5
        ok, reason = self.validate(text)
        self.assertTrue(ok)

    def test_pdf_residue_rejected(self):
        text = "Some text >>stream /Filter /FlateDecode endobj more text " * 5
        ok, reason = self.validate(text)
        self.assertFalse(ok)
        self.assertIn("pdf_binary_residue", reason)

    def test_single_pdf_marker_ok(self):
        text = (
            "This paper discusses stream processing in distributed systems. "
            "We evaluate our approach using multiple benchmarks. "
        ) * 10
        ok, reason = self.validate(text)
        self.assertTrue(ok)


class TestIsPaywallGarbage(unittest.TestCase):

    def setUp(self):
        from autonomy.knowledge_integrator import _is_paywall_garbage
        self.check = _is_paywall_garbage

    def test_obvious_paywall_detected(self):
        text = (
            "Sign in to access this article. Create account to continue. "
            "Purchase details available after login. "
        ) + "word " * 60
        self.assertTrue(self.check(text))

    def test_short_text_rejected(self):
        self.assertTrue(self.check("Just a few words here"))

    def test_real_abstract_passes(self):
        text = (
            "Abstract: We introduce a novel method for semantic segmentation "
            "that combines convolutional neural networks with attention mechanisms. "
            "Our experimental results on PASCAL VOC and Cityscapes demonstrate "
            "state-of-the-art performance. The proposed approach uses a multi-scale "
            "feature extraction module followed by a self-attention layer. "
            "We evaluate on standard benchmarks and report improvements of 2.3% "
            "mIoU over the previous best method. "
        ) * 2
        self.assertFalse(self.check(text))

    def test_cookie_consent_page_detected(self):
        text = (
            "We use cookies to improve your experience. Cookie policy details. "
            "Accept cookies or manage preferences to continue browsing. "
            "Your privacy choices matter to us. "
        ) + "word " * 60
        self.assertTrue(self.check(text))

    def test_boilerplate_dominant_page_detected(self):
        text = (
            "Cookie preferences. Privacy policy. Terms of service. "
            "Subscribe to our newsletter. Navigation menu sidebar footer. "
            "All rights reserved copyright 2024. "
        ) + "word " * 60
        self.assertTrue(self.check(text))

    def test_paywall_with_academic_keyword_passes_if_dominant(self):
        text = (
            "Abstract: This paper presents novel results in machine learning. "
            "Introduction: We discuss the methodology and experimental evaluation. "
            "Method: Our approach uses a transformer-based architecture. "
            "Results: We achieve 95% accuracy on the test set with our proposed framework. "
            "Conclusion: The experimental results demonstrate the effectiveness. "
        ) * 3
        self.assertFalse(self.check(text))

    def test_no_academic_signals_with_one_paywall_marker(self):
        text = (
            "Sign in to view this content. This page contains various "
            "information about products and services that might interest you. "
        ) + "word " * 60
        self.assertTrue(self.check(text))


class TestScoreAcademicContent(unittest.TestCase):

    def setUp(self):
        from autonomy.knowledge_integrator import _score_academic_content
        self.score = _score_academic_content

    def test_academic_paper_scores_high(self):
        text = (
            "Abstract Introduction Method Results Discussion Conclusion "
            "We propose a novel evaluation framework with baseline ablation "
            "on a standard dataset."
        )
        academic, boilerplate = self.score(text)
        self.assertGreater(academic, boilerplate)
        self.assertGreater(academic, 5)

    def test_landing_page_scores_boilerplate(self):
        text = (
            "Cookie privacy terms of sign in log in subscribe newsletter "
            "copyright all rights reserved javascript your browser enable cookies "
            "accept all manage preferences navigation menu sidebar footer"
        )
        academic, boilerplate = self.score(text)
        self.assertGreater(boilerplate, academic)

    def test_empty_text_scores_zero(self):
        academic, boilerplate = self.score("")
        self.assertEqual(academic, 0)
        self.assertEqual(boilerplate, 0)


class TestExtractPdfText(unittest.TestCase):

    def setUp(self):
        from library.ingest import _extract_pdf_text
        self.extract = _extract_pdf_text

    @patch("subprocess.run")
    def test_successful_extraction(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"This is extracted PDF text. " * 20,
        )
        text, err = self.extract(b"%PDF-1.4 fake pdf data")
        self.assertTrue(len(text) > 100)
        self.assertEqual(err, "")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_pdftotext_not_installed(self, mock_run):
        text, err = self.extract(b"%PDF-1.4 data")
        self.assertEqual(text, "")
        self.assertEqual(err, "pdftotext_not_installed")

    @patch("subprocess.run")
    def test_short_extraction_rejected(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"Short",
        )
        text, err = self.extract(b"%PDF-1.4 data")
        self.assertEqual(text, "")
        self.assertEqual(err, "pdf_extraction_too_short")


class TestFetchUrlContentTypeRouting(unittest.TestCase):

    def test_unsupported_content_type_rejected(self):
        from library.ingest import _fetch_url

        with patch("library.ingest._validate_url", return_value=None):
            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Type": "image/png"}
            mock_resp.read.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                text, err = _fetch_url("https://example.com/image.png")
                self.assertEqual(text, "")
                self.assertIn("unsupported_content_type", err)

    def test_html_entities_decoded(self):
        from library.ingest import _strip_html
        result = _strip_html("<p>Hello &amp; World &lt;3</p>")
        self.assertIn("Hello & World <3", result)


class TestHtmlEntityDecoding(unittest.TestCase):

    def test_unescape_common_entities(self):
        text = "R&amp;D in AI &lt;2024&gt; with &quot;transformers&quot;"
        result = html.unescape(text)
        self.assertIn("R&D", result)
        self.assertIn("<2024>", result)
        self.assertIn('"transformers"', result)


if __name__ == "__main__":
    unittest.main()
