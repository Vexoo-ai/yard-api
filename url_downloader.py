"""
url_downloader.py

Standalone URL downloader utility for fetching documents from public URLs.

Supports:
- Direct document URLs (PDF, DOCX, XLSX, PPTX, TXT, CSV, HTML, images, etc.)
- ZIP archives (auto-extracted, all supported files saved)
- Google Drive file sharing links (file/d/ID and open?id=ID formats)
- Google Drive folder links (entire folder downloaded via gdown)
- Dropbox sharing links (dl=0 converted to dl=1)
- OneDrive sharing links (download=1 appended)
- SharePoint links (download=1 appended)

Usage:
    from url_downloader import download_urls

    local_paths = await download_urls(
        urls=["https://...", "https://..."],
        target_dir="/tmp/my_session_docs"
    )
    # local_paths is a list of absolute file paths ready for processing
"""

import asyncio
import logging
import os
import re
import uuid
import zipfile
import io
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse, unquote

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file extensions
# Must match what DocumentProcessor / main.py already accepts
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.txt', '.md', '.markdown',
    '.pptx', '.ppt', '.xlsx', '.xls', '.csv',
    '.html', '.htm',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif',
}

# ZIP is a container — we unpack it, not process it directly
CONTAINER_EXTENSIONS = {'.zip'}

# Content-Type → extension mapping (used when URL/headers don't reveal type)
CONTENT_TYPE_TO_EXT = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.ms-powerpoint': '.ppt',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/csv': '.csv',
    'text/html': '.html',
    'text/xml': '.xml',
    'application/xml': '.xml',
    'application/json': '.json',
    'application/epub+zip': '.epub',
    'application/rtf': '.rtf',
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'application/zip': '.zip',
    'application/x-zip-compressed': '.zip',
    'application/x-zip': '.zip',
    'multipart/x-zip': '.zip',
}

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB hard limit per file


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

async def download_urls(urls: List[str], target_dir: str) -> List[str]:
    """
    Download documents from a list of public URLs into target_dir.

    Handles Google Drive files/folders, Dropbox, OneDrive, SharePoint,
    direct document links, and ZIP archives.

    Args:
        urls:       List of public URLs to download from.
        target_dir: Local directory where downloaded files will be saved.
                    Will be created if it does not exist.

    Returns:
        List of absolute paths to all successfully downloaded/extracted files.
        Files that failed to download are omitted (errors are logged).
    """
    os.makedirs(target_dir, exist_ok=True)

    tasks = [_download_single_url(url, target_dir) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_paths: List[str] = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.error(f"Unhandled exception downloading {url}: {result}")
        elif result["success"]:
            all_paths.extend(result["local_paths"])
            logger.info(
                f"Downloaded {url} -> {len(result['local_paths'])} file(s)"
            )
        else:
            logger.warning(
                f"Failed to download {url}: {result.get('error', 'unknown error')}"
            )

    return all_paths


# ===========================================================================
# URL RESOLVER
# Converts sharing/viewing links to direct download URLs.
# Returns (resolved_url, provider_tag)
# provider_tag is one of: "google_drive_folder", "google_drive",
#                          "dropbox", "onedrive", "sharepoint", None
# ===========================================================================

def _resolve_url(original_url: str) -> Tuple[str, Optional[str]]:
    """
    Inspect a URL and convert cloud-storage sharing links to direct
    download URLs.

    Google Drive folder links are returned as a sentinel
    ``gdrive_folder://<FOLDER_ID>`` so the caller can route them to
    gdown instead of aiohttp.

    Returns:
        (resolved_url, provider_tag)
    """

    # ── Google Drive FOLDER ─────────────────────────────────────────────
    # https://drive.google.com/drive/folders/FOLDER_ID
    gd_folder = re.search(
        r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
        original_url
    )
    if gd_folder:
        folder_id = gd_folder.group(1)
        logger.info(f"Google Drive folder URL detected. Folder ID: {folder_id}")
        return f"gdrive_folder://{folder_id}", "google_drive_folder"

    # ── Google Drive FILE (file/d/ID/view) ──────────────────────────────
    gd_file = re.search(
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        original_url
    )
    if gd_file:
        file_id = gd_file.group(1)
        direct = (
            f"https://drive.google.com/uc"
            f"?export=download&id={file_id}&confirm=t"
        )
        logger.info(f"Google Drive file URL detected. Resolved: {direct}")
        return direct, "google_drive"

    # ── Google Drive OPEN LINK (open?id=ID) ─────────────────────────────
    gd_open = re.search(
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        original_url
    )
    if gd_open:
        file_id = gd_open.group(1)
        direct = (
            f"https://drive.google.com/uc"
            f"?export=download&id={file_id}&confirm=t"
        )
        logger.info(f"Google Drive open URL detected. Resolved: {direct}")
        return direct, "google_drive"

    # ── Dropbox ──────────────────────────────────────────────────────────
    if 'dropbox.com' in original_url:
        resolved = re.sub(
            r'[?&]dl=0',
            lambda m: m.group(0).replace('dl=0', 'dl=1'),
            original_url
        )
        if 'dl=1' not in resolved:
            sep = '&' if '?' in resolved else '?'
            resolved = f"{resolved}{sep}dl=1"
        logger.info(f"Dropbox URL detected. Resolved: {resolved}")
        return resolved, "dropbox"

    # ── OneDrive ─────────────────────────────────────────────────────────
    if 'onedrive.live.com' in original_url or '1drv.ms' in original_url:
        resolved = original_url
        if 'download=1' not in resolved:
            sep = '&' if '?' in resolved else '?'
            resolved = f"{resolved}{sep}download=1"
        logger.info(f"OneDrive URL detected. Resolved: {resolved}")
        return resolved, "onedrive"

    # ── SharePoint ───────────────────────────────────────────────────────
    if 'sharepoint.com' in original_url:
        resolved = original_url
        if 'download=1' not in resolved:
            sep = '&' if '?' in resolved else '?'
            resolved = f"{resolved}{sep}download=1"
        logger.info(f"SharePoint URL detected. Resolved: {resolved}")
        return resolved, "sharepoint"

    # ── Plain URL (no conversion needed) ────────────────────────────────
    return original_url, None


# ===========================================================================
# FILENAME DERIVER
# Priority: Content-Disposition header > URL path > Content-Type
# ===========================================================================

def _derive_filename(
    url: str,
    content_type: str = "",
    content_disposition: str = ""
) -> Tuple[str, Optional[str]]:
    """
    Derive a safe filename and its extension from response metadata.

    Returns:
        (filename_with_uuid_suffix, extension_or_None)
    """
    all_extensions = SUPPORTED_EXTENSIONS | CONTAINER_EXTENSIONS

    # 1. Content-Disposition header
    if content_disposition:
        cd_match = re.search(
            r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)["\']?',
            content_disposition,
            re.IGNORECASE
        )
        if cd_match:
            cd_name = unquote(cd_match.group(1).strip())
            base, ext = os.path.splitext(cd_name)
            ext = ext.lower()
            if ext in all_extensions:
                safe_base = re.sub(r'[^\w\-.]', '_', base)[:80]
                return f"{safe_base}_{uuid.uuid4().hex[:8]}{ext}", ext

    # 2. URL path
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        raw_name = os.path.basename(path).split('?')[0].strip()
        if raw_name:
            base, ext = os.path.splitext(raw_name)
            ext = ext.lower()
            if ext in all_extensions:
                safe_base = re.sub(r'[^\w\-.]', '_', base)[:80]
                return f"{safe_base}_{uuid.uuid4().hex[:8]}{ext}", ext
    except Exception:
        pass

    # 3. Content-Type fallback
    if content_type:
        ct_base = content_type.split(';')[0].strip().lower()
        ext = CONTENT_TYPE_TO_EXT.get(ct_base)
        if ext:
            return f"document_{uuid.uuid4().hex[:12]}{ext}", ext

    # 4. Unknown — caller will try magic bytes
    return f"document_{uuid.uuid4().hex[:12]}", None


# ===========================================================================
# MAGIC BYTES DETECTION
# Last resort when headers/URL give no type information
# ===========================================================================

def _detect_extension_from_magic(content: bytes, content_type: str = "") -> Optional[str]:
    """
    Inspect the first few bytes of file content to determine file type.

    Returns extension string (e.g. '.pdf') or None if undetectable.
    """
    magic = content[:8]

    if magic[:4] == b'%PDF':
        return '.pdf'

    if magic[:4] == b'PK\x03\x04':
        # ZIP-based format — narrow down by content-type if available
        ct = content_type.lower()
        if 'wordprocessingml' in ct:
            return '.docx'
        if 'spreadsheetml' in ct:
            return '.xlsx'
        if 'presentationml' in ct:
            return '.pptx'
        return '.zip'

    # UTF BOM markers → likely a text file
    if magic[:3] in (b'\xef\xbb\xbf',) or magic[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return '.txt'

    # PNG signature
    if magic[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'

    # JPEG
    if magic[:2] == b'\xff\xd8':
        return '.jpg'

    # GIF
    if magic[:6] in (b'GIF87a', b'GIF89a'):
        return '.gif'

    return None


# ===========================================================================
# GOOGLE DRIVE FOLDER DOWNLOADER (uses gdown)
# ===========================================================================

async def _download_gdrive_folder(
    folder_url: str,
    folder_id: str,
    target_dir: str
) -> dict:
    """
    Download all supported files from a public Google Drive folder using gdown.

    gdown is a synchronous library, so we run it in a thread pool executor.

    Returns:
        {success, local_paths, error}
    """
    try:
        import gdown
    except ImportError:
        return {
            "success": False,
            "local_paths": [],
            "error": (
                "gdown is not installed. "
                "Add 'gdown>=5.1.0' to requirements.txt and reinstall."
            )
        }

    import tempfile

    local_paths: List[str] = []

    try:
        loop = asyncio.get_event_loop()

        # gdown downloads into a temp directory; we then filter and copy
        with tempfile.TemporaryDirectory() as tmp_dir:

            def _run_gdown():
                return gdown.download_folder(
                    url=folder_url,
                    output=tmp_dir,
                    quiet=True,
                    use_cookies=False,
                    remaining_ok=True,
                )

            try:
                downloaded_paths = await loop.run_in_executor(None, _run_gdown)
            except Exception as e:
                return {
                    "success": False,
                    "local_paths": [],
                    "error": (
                        f"gdown failed. Ensure the folder is publicly shared "
                        f"('Anyone with the link'). Error: {e}"
                    )
                }

            if not downloaded_paths:
                return {
                    "success": False,
                    "local_paths": [],
                    "error": (
                        "No files were downloaded. Ensure the folder is "
                        "publicly shared and not empty."
                    )
                }

            used_names = set()

            for downloaded_path in downloaded_paths:
                if not os.path.isfile(downloaded_path):
                    continue

                basename = os.path.basename(downloaded_path)
                _, ext = os.path.splitext(basename)
                ext = ext.lower()

                # Skip hidden / system files
                if basename.startswith('.') or basename.startswith('~$'):
                    logger.debug(f"GDrive folder: skipping hidden file {basename}")
                    continue

                # Skip unsupported extensions
                if ext not in SUPPORTED_EXTENSIONS:
                    logger.debug(
                        f"GDrive folder: skipping unsupported file {basename} (ext={ext})"
                    )
                    continue

                # Build a flat unique filename
                try:
                    rel_path = os.path.relpath(downloaded_path, tmp_dir)
                    parts = rel_path.replace("\\", "/").split("/")
                    folder_parts = parts[:-1]
                    base_no_ext = os.path.splitext(basename)[0]
                    combined = "_".join(folder_parts + [base_no_ext]) if folder_parts else base_no_ext
                    safe = re.sub(r'[^\w\-]', '_', combined)
                    safe = re.sub(r'_+', '_', safe).strip('_')[:100]
                    flat_name = f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
                    while flat_name in used_names:
                        flat_name = f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
                    used_names.add(flat_name)
                except Exception:
                    flat_name = f"gdrive_{uuid.uuid4().hex[:12]}{ext}"
                    used_names.add(flat_name)

                dest_path = os.path.join(target_dir, flat_name)
                with open(downloaded_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())

                local_paths.append(dest_path)
                logger.info(
                    f"GDrive folder: saved '{basename}' -> '{flat_name}'"
                )

        if not local_paths:
            return {
                "success": False,
                "local_paths": [],
                "error": (
                    "No supported files found in Google Drive folder. "
                    f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            }

        return {"success": True, "local_paths": local_paths, "error": None}

    except Exception as e:
        logger.error(f"GDrive folder download error for {folder_url}: {e}")
        return {"success": False, "local_paths": [], "error": str(e)}


# ===========================================================================
# ZIP EXTRACTOR
# ===========================================================================

def _extract_zip_to_dir(
    zip_content: bytes,
    target_dir: str,
    source_label: str = "zip"
) -> List[str]:
    """
    Extract a ZIP archive and save all supported documents into target_dir.

    Handles nested folders (flattened into target_dir), Windows backslash
    paths, macOS metadata files, hidden files, and zero-size entries.

    Returns:
        List of absolute paths of successfully extracted files.
    """

    def _normalize(raw_path: str) -> str:
        return raw_path.replace('\\', '/')

    def _is_system_file(normalized_path: str, basename: str) -> bool:
        parts = normalized_path.split('/')
        if '__MACOSX' in parts:
            return True
        if basename.startswith('.') or basename.startswith('~$'):
            return True
        if basename.lower() in ('thumbs.db', 'desktop.ini'):
            return True
        if ':Zone.Identifier' in basename:
            return True
        return False

    if not zipfile.is_zipfile(io.BytesIO(zip_content)):
        raise ValueError("Content is not a valid ZIP archive.")

    extracted_paths: List[str] = []
    used_names: set = set()

    with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
        all_members = zf.infolist()
        logger.info(
            f"ZIP ({source_label}): {len(all_members)} total member(s)"
        )

        for member in all_members:
            normalized_path = _normalize(member.filename)
            basename = os.path.basename(normalized_path)

            # Skip directories
            if (
                member.is_dir()
                or normalized_path.endswith('/')
                or not basename
                or not basename.strip()
            ):
                continue

            # Skip system/hidden files
            if _is_system_file(normalized_path, basename):
                logger.debug(f"ZIP: skipping system file {member.filename}")
                continue

            # Extension check
            _, ext = os.path.splitext(basename)
            ext = ext.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                logger.debug(
                    f"ZIP: skipping unsupported file {member.filename} (ext={ext})"
                )
                continue

            # Skip zero-size entries
            if member.file_size == 0:
                logger.debug(f"ZIP: skipping empty file {member.filename}")
                continue

            # Build flat unique filename (preserve folder context as prefix)
            parts = normalized_path.split('/')
            folder_parts = parts[:-1]
            base_no_ext = os.path.splitext(basename)[0]
            combined = "_".join(folder_parts + [base_no_ext]) if folder_parts else base_no_ext
            safe = re.sub(r'[^\w\-]', '_', combined)
            safe = re.sub(r'_+', '_', safe).strip('_')[:100]
            flat_name = f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
            while flat_name in used_names:
                flat_name = f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
            used_names.add(flat_name)

            # Extract and write
            try:
                file_data = zf.read(member.filename)
                if len(file_data) == 0:
                    continue
                dest_path = os.path.join(target_dir, flat_name)
                with open(dest_path, 'wb') as f:
                    f.write(file_data)
                extracted_paths.append(dest_path)
                logger.info(
                    f"ZIP: extracted '{member.filename}' -> '{flat_name}' "
                    f"({len(file_data)} bytes)"
                )
            except zipfile.BadZipFile as e:
                logger.error(f"ZIP: bad entry '{member.filename}': {e}")
            except Exception as e:
                logger.error(f"ZIP: error extracting '{member.filename}': {e}")

    return extracted_paths


# ===========================================================================
# SINGLE URL DOWNLOADER
# Orchestrates resolve → download → detect type → extract if ZIP
# ===========================================================================

async def _download_single_url(url: str, target_dir: str) -> dict:
    """
    Download a single URL and save the result(s) to target_dir.

    Returns:
        {success, local_paths, error}
        local_paths is a list because ZIPs and GDrive folders expand to many files.
    """
    original_url = url

    try:
        resolved_url, cloud_provider = _resolve_url(url)

        # ── Google Drive FOLDER ──────────────────────────────────────────
        if cloud_provider == "google_drive_folder":
            folder_id = resolved_url.replace("gdrive_folder://", "")
            return await _download_gdrive_folder(
                folder_url=url,
                folder_id=folder_id,
                target_dir=target_dir
            )

        # ── Standard HTTP download ───────────────────────────────────────
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        filename = None
        detected_ext = None
        content_type = ""
        content_disposition = ""

        async with aiohttp.ClientSession() as session:

            # Step 1: HEAD request (best-effort — some servers don't support it)
            try:
                async with session.head(
                    resolved_url,
                    ssl=ssl_context,
                    timeout=aiohttp.ClientTimeout(total=15),
                    allow_redirects=True
                ) as head_resp:
                    content_type = head_resp.headers.get("Content-Type", "")
                    content_disposition = head_resp.headers.get("Content-Disposition", "")
                    content_length = head_resp.headers.get("Content-Length")

                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        size_mb = int(content_length) / (1024 * 1024)
                        return {
                            "success": False,
                            "local_paths": [],
                            "error": (
                                f"File too large: {size_mb:.1f} MB. "
                                f"Maximum allowed: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB."
                            )
                        }
            except aiohttp.ClientError as e:
                logger.warning(
                    f"HEAD request failed for {resolved_url}: {e}. "
                    f"Proceeding with GET."
                )

            # Attempt filename derivation from HEAD info
            filename, detected_ext = _derive_filename(
                resolved_url, content_type, content_disposition
            )

            # Step 2: GET — download the actual file
            async with session.get(
                resolved_url,
                ssl=ssl_context,
                timeout=aiohttp.ClientTimeout(total=300),
                allow_redirects=True
            ) as resp:
                if resp.status != 200:
                    return {
                        "success": False,
                        "local_paths": [],
                        "error": (
                            f"HTTP {resp.status} when downloading {original_url}. "
                            f"Verify the URL is publicly accessible."
                        )
                    }

                # Refine filename from GET response headers (may be richer)
                get_content_type = resp.headers.get("Content-Type", "")
                get_content_disposition = resp.headers.get("Content-Disposition", "")
                if get_content_type or get_content_disposition:
                    better_filename, better_ext = _derive_filename(
                        resolved_url, get_content_type, get_content_disposition
                    )
                    if better_ext:
                        filename = better_filename
                        detected_ext = better_ext
                        content_type = get_content_type

                # Read content with size guard
                content = bytearray()
                async for chunk in resp.content.iter_chunked(65536):
                    content.extend(chunk)
                    if len(content) > MAX_FILE_SIZE:
                        return {
                            "success": False,
                            "local_paths": [],
                            "error": (
                                f"File exceeds maximum size of "
                                f"{MAX_FILE_SIZE / (1024 * 1024):.0f} MB."
                            )
                        }

                content = bytes(content)
                size_bytes = len(content)

                if size_bytes < 10:
                    return {
                        "success": False,
                        "local_paths": [],
                        "error": (
                            f"Downloaded file is too small ({size_bytes} bytes). "
                            f"The URL may not point to a valid document."
                        )
                    }

                # ── Google Drive virus-scan confirmation page ─────────────
                # GDrive sometimes redirects large files to a warning HTML page.
                # We detect this and re-request with the confirm token.
                if (
                    cloud_provider == "google_drive"
                    and b"<html" in content[:500].lower()
                ):
                    html_str = content.decode('utf-8', errors='ignore')
                    confirm_match = re.search(
                        r'confirm=([0-9A-Za-z_\-]+)', html_str
                    )
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        gd_file_match = re.search(
                            r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
                            original_url
                        )
                        if gd_file_match:
                            file_id = gd_file_match.group(1)
                            confirm_url = (
                                f"https://drive.google.com/uc"
                                f"?export=download&id={file_id}"
                                f"&confirm={confirm_token}"
                            )
                            logger.info(
                                f"GDrive virus-scan page detected. "
                                f"Re-downloading with confirm token."
                            )
                            async with session.get(
                                confirm_url,
                                ssl=ssl_context,
                                timeout=aiohttp.ClientTimeout(total=300),
                                allow_redirects=True
                            ) as confirm_resp:
                                if confirm_resp.status == 200:
                                    content = await confirm_resp.read()
                                    # Update filename from confirm response
                                    conf_ct = confirm_resp.headers.get("Content-Type", "")
                                    conf_cd = confirm_resp.headers.get("Content-Disposition", "")
                                    if conf_ct or conf_cd:
                                        better_fn, better_ext = _derive_filename(
                                            confirm_url, conf_ct, conf_cd
                                        )
                                        if better_ext:
                                            filename = better_fn
                                            detected_ext = better_ext
                                            content_type = conf_ct

                # ── Reject HTML pages masquerading as documents ───────────
                final_ct = get_content_type.lower()
                if (
                    "text/html" in final_ct
                    and detected_ext not in ('.html', '.htm')
                    and b"<html" in content[:500].lower()
                ):
                    return {
                        "success": False,
                        "local_paths": [],
                        "error": (
                            f"URL returned an HTML page instead of a document. "
                            f"The link may require login or may not be a direct download URL."
                        )
                    }

                # ── Magic bytes detection if extension still unknown ───────
                if not detected_ext or detected_ext not in (SUPPORTED_EXTENSIONS | CONTAINER_EXTENSIONS):
                    detected_ext = _detect_extension_from_magic(content, get_content_type)
                    if detected_ext:
                        filename = f"document_{uuid.uuid4().hex[:12]}{detected_ext}"
                    else:
                        return {
                            "success": False,
                            "local_paths": [],
                            "error": (
                                f"Could not determine document type for {original_url}. "
                                f"Content-Type: '{get_content_type}'. "
                                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                            )
                        }

        # ── ZIP handling ─────────────────────────────────────────────────
        is_zip = (
            detected_ext == '.zip'
            or (
                content[:4] == b'PK\x03\x04'
                and detected_ext not in ('.docx', '.xlsx', '.pptx')
            )
        )

        if is_zip:
            logger.info(f"ZIP archive detected from {original_url}. Extracting...")
            try:
                extracted = _extract_zip_to_dir(
                    zip_content=content,
                    target_dir=target_dir,
                    source_label=original_url
                )
            except ValueError as e:
                return {
                    "success": False,
                    "local_paths": [],
                    "error": str(e)
                }

            if not extracted:
                return {
                    "success": False,
                    "local_paths": [],
                    "error": (
                        f"ZIP from {original_url} contained no supported files. "
                        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                    )
                }

            return {"success": True, "local_paths": extracted, "error": None}

        # ── Regular document — save directly ─────────────────────────────
        if detected_ext not in SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "local_paths": [],
                "error": (
                    f"Extension '{detected_ext}' is not supported. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            }

        file_path = os.path.join(target_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(
            f"Downloaded: {original_url} -> {filename} "
            f"({size_bytes} bytes, ext={detected_ext}, provider={cloud_provider or 'direct'})"
        )

        return {"success": True, "local_paths": [file_path], "error": None}

    except Exception as e:
        logger.error(f"Error downloading {original_url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "local_paths": [], "error": str(e)}
