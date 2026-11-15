"""Application error types.

Every failure the UI should present to a user is an ``AppError`` subclass with a
message written for humans. Anything else is a bug and should surface loudly.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-presentable failures."""


class ConfigError(AppError):
    """Missing or invalid configuration (usually environment variables)."""


class PdfError(AppError):
    """Base class for problems reading a PDF."""


class EncryptedPdfError(PdfError):
    """The PDF is password protected and cannot be opened."""


class PdfTooLargeError(PdfError):
    """The PDF has more pages than the configured guardrail allows."""


class EmptyDocumentError(PdfError):
    """The PDF has no usable text layer (typically a scan)."""


class SummarizationError(AppError):
    """The model could not produce a usable summary."""

