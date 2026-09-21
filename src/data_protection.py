"""User-bound at-rest protection for biometric cache data."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from typing import Protocol


class DataProtectionError(RuntimeError):
    """Raised when biometric data cannot be protected or recovered."""


class DataProtector(Protocol):
    def protect(self, data: bytes) -> bytes: ...

    def unprotect(self, data: bytes) -> bytes: ...


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _make_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)),
    )
    return blob, buffer


class WindowsDpapiProtector:
    """Encrypt data so only the current Windows user can decrypt it."""

    _entropy = b"face-recognition-system:encoding-cache:v1"
    _ui_forbidden = 0x1

    def __init__(self) -> None:
        if os.name != "nt":
            raise DataProtectionError("Windows DPAPI is only available on Windows.")
        self.crypt32 = ctypes.windll.crypt32
        self.kernel32 = ctypes.windll.kernel32
        self.crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            wintypes.LPCWSTR,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        self.crypt32.CryptProtectData.restype = wintypes.BOOL
        self.crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        self.crypt32.CryptUnprotectData.restype = wintypes.BOOL
        self.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel32.LocalFree.restype = ctypes.c_void_p

    def protect(self, data: bytes) -> bytes:
        input_blob, input_buffer = _make_blob(data)
        entropy_blob, entropy_buffer = _make_blob(self._entropy)
        output_blob = _DataBlob()
        success = self.crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "Face recognition encoding cache",
            ctypes.byref(entropy_blob),
            None,
            None,
            self._ui_forbidden,
            ctypes.byref(output_blob),
        )
        if not success:
            raise DataProtectionError(f"DPAPI encryption failed: {ctypes.WinError()}")
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            self.kernel32.LocalFree(output_blob.pbData)

    def unprotect(self, data: bytes) -> bytes:
        input_blob, input_buffer = _make_blob(data)
        entropy_blob, entropy_buffer = _make_blob(self._entropy)
        output_blob = _DataBlob()
        success = self.crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            ctypes.byref(entropy_blob),
            None,
            None,
            self._ui_forbidden,
            ctypes.byref(output_blob),
        )
        if not success:
            raise DataProtectionError(f"DPAPI decryption failed: {ctypes.WinError()}")
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            self.kernel32.LocalFree(output_blob.pbData)


def default_data_protector() -> DataProtector:
    if os.name == "nt":
        return WindowsDpapiProtector()
    raise DataProtectionError(
        "Encrypted cache protection is not configured for this operating system. "
        "Use --disable-cache-encryption only for non-sensitive development."
    )
