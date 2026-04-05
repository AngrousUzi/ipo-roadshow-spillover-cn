"""
analyze/initialize/ffmpeg_utils.py
===================================
Low-level FFmpeg helpers: integrity checking, quality probing,
video clipping, concatenation (stream-copy and re-encode), and
timestamp-accurate cutting.

All functions are pure (no global state) and safe to call from
multiprocessing workers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import ffmpeg


# ── Integrity & Quality ────────────────────────────────────────────────────────

def check_video_integrity(video_path: Path) -> str:
    """
    Run ``ffmpeg -v error -i <video> -c copy -f null -`` to validate the
    container.  Returns stderr content; empty string means no errors.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path), "-c", "copy", "-f", "null", "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        err = result.stderr.strip()
        if err:
            print(f"[WARN] 视频完整性检查报错: {video_path}\n{err}")
        return err
    except Exception as e:
        msg = f"完整性检查异常: {e}"
        print(f"[WARN] {msg}")
        return msg


def check_video_quality(video_path: Path, full_decode: bool = True) -> dict:
    """
    Probe metadata with ffprobe and optionally full-decode to detect frame
    errors.

    Returned keys
    -------------
    index2009, filename,
    codec_video, width, height, fps,
    video_bitrate_kbps, duration_sec, container_duration_sec,
    has_audio, audio_codec,
    integrity_error, decode_error_count, decode_error_sample,
    quality_flags, gaze_flags
    """
    stem = video_path.stem
    index2009 = stem.split("_")[0] if "_" in stem else stem

    record: dict = {
        "index2009":              index2009,
        "filename":               video_path.name,
        "codec_video":            "",
        "width":                  None,
        "height":                 None,
        "fps":                    None,
        "video_bitrate_kbps":     None,
        "duration_sec":           None,
        "container_duration_sec": None,
        "has_audio":              False,
        "audio_codec":            "",
        "integrity_error":        "",
        "decode_error_count":     0,
        "decode_error_sample":    "",
        "quality_flags":          "",
        "gaze_flags":             "",
    }

    # 1. ffprobe metadata
    try:
        probe_result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(video_path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        probe = json.loads(probe_result.stdout)
    except Exception as e:
        record["quality_flags"] = f"ffprobe失败: {e}"
        return record

    fmt = probe.get("format", {})
    container_dur = fmt.get("duration")
    record["container_duration_sec"] = round(float(container_dur), 2) if container_dur else None

    for stream in probe.get("streams", []):
        ctype = stream.get("codec_type", "")
        if ctype == "video" and not record["codec_video"]:
            record["codec_video"] = stream.get("codec_name", "")
            record["width"]       = stream.get("width")
            record["height"]      = stream.get("height")
            r_fps = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_fps.split("/")
                record["fps"] = round(int(num) / int(den), 2) if int(den) else None
            except Exception:
                record["fps"] = None
            br = stream.get("bit_rate") or fmt.get("bit_rate")
            record["video_bitrate_kbps"] = round(int(br) / 1000, 1) if br else None
            dur = stream.get("duration") or fmt.get("duration")
            record["duration_sec"] = round(float(dur), 2) if dur else None
        elif ctype == "audio" and not record["has_audio"]:
            record["has_audio"]   = True
            record["audio_codec"] = stream.get("codec_name", "")

    # 2. Container integrity
    record["integrity_error"] = check_video_integrity(video_path)

    # 3. Full-decode error detection
    if full_decode:
        try:
            dec = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore",
            )
            err_lines = [ln for ln in dec.stderr.splitlines() if ln.strip()]
            record["decode_error_count"]  = len(err_lines)
            record["decode_error_sample"] = " | ".join(err_lines[:3])
        except Exception as e:
            record["decode_error_count"]  = -1
            record["decode_error_sample"] = str(e)

    # 4. Quality flags
    flags: list[str] = []
    br = record["video_bitrate_kbps"]
    if br is not None and br < 100:
        flags.append(f"码率极低({br}kbps)")
    dur = record["duration_sec"]
    if dur is not None and dur < 60:
        flags.append(f"时长过短({dur}s)")
    w, h = record["width"], record["height"]
    if w and h and w * h < 960 * 540:
        flags.append(f"分辨率低于540p({w}x{h})")
    if not record["has_audio"]:
        flags.append("无音频流")
    if full_decode and record["decode_error_count"] > 0:
        flags.append(f"解码错误({record['decode_error_count']}行)")
    if record["integrity_error"]:
        flags.append("容器完整性异常")
    cdr = record["container_duration_sec"]
    if dur and cdr and abs(dur - cdr) > 2:
        flags.append(f"时长不一致(差{abs(dur - cdr):.1f}s)")
    record["quality_flags"] = "; ".join(flags)

    # 5. Gaze analysis hints
    gaze_notes: list[str] = []
    if record["fps"] is not None and record["fps"] < 12:
        gaze_notes.append(f"帧率{record['fps']}fps低于VISUAL_SAMPLE_FPS=12")
    if record["height"] is not None and record["height"] < 480:
        gaze_notes.append(f"分辨率{record['width']}x{record['height']}偏低")
    if br is not None and br < 500 and (record["height"] or 0) >= 720:
        gaze_notes.append(f"720p+视频码率仅{br}kbps")
    record["gaze_flags"] = "; ".join(gaze_notes)

    return record


# ── Clip (single-file copy) ────────────────────────────────────────────────────

def clip_video(video_path: Path, output_path: Path) -> tuple[Path, str]:
    """
    Copy *video_path* to *output_path* and run an integrity check.
    Returns ``(output_path, error_str)``; empty *error_str* = success.
    """
    if output_path.exists():
        return output_path, check_video_integrity(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, output_path)
    return output_path, check_video_integrity(output_path)


# ── Cut (timestamp-accurate segment extraction) ────────────────────────────────

def cut_video(
    source: Path,
    output_path: Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[bool, str]:
    """
    Extract a segment ``[start_sec, end_sec)`` from *source* using stream
    copy (fast, keyframe-aligned).

    When *end_sec* is ``None`` the cut runs to the end of the file.
    Returns ``(success, error_str)``.
    """
    if output_path.exists():
        err = check_video_integrity(output_path)
        return err == "", err

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".cut.tmp" + output_path.suffix)

    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(source)]
    if end_sec is not None:
        # -to is relative to the re-based timeline after input seeking
        cmd += ["-to", str(end_sec - start_sec)]
    cmd += ["-c", "copy", str(temp_path)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        error_str = result.stderr.strip()
        if result.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"cut失败(exit {result.returncode}): {error_str}"
        temp_path.replace(output_path)
        return True, error_str
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, f"cut异常: {e}"


# ── Concat: stream-copy ────────────────────────────────────────────────────────

def concat_videos(video_paths: list[Path], output_path: Path) -> tuple[bool, str]:
    """
    Concatenate videos using the FFmpeg concat demuxer (stream copy, fast).
    Returns ``(success, error_str)``.
    """
    if not video_paths:
        return False, "[ERROR] 未提供视频路径"

    normalized = [Path(p) for p in video_paths]
    missing = [str(p) for p in normalized if not p.exists()]
    if missing:
        return False, f"[ERROR] 文件不存在: {missing}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    list_file: Path | None = None
    error_str = ""

    try:
        if temp_path.exists():
            temp_path.unlink()

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for p in normalized:
                escaped = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            list_file = Path(f.name)

        _, err = (
            ffmpeg
            .input(str(list_file), format="concat", safe=0)
            .output(str(temp_path), c="copy")
            .global_args("-loglevel", "error")
            .overwrite_output()
            .run(capture_stderr=True)
        )
        if err:
            decoded = err.decode("utf8", errors="ignore").strip()
            if decoded:
                print(f"{output_path.name}: FFmpeg stderr: {decoded}")
                error_str = decoded

        temp_path.replace(output_path)
        return True, error_str

    except ffmpeg.Error as e:
        lines = ["FFmpeg concat失败:"]
        if e.stderr:
            lines.append(e.stderr.decode("utf8", errors="ignore"))
        msg = "\n".join(lines)
        if temp_path.exists():
            temp_path.unlink()
        return False, msg
    except Exception as e:
        msg = f"concat异常: {e}"
        if temp_path.exists():
            temp_path.unlink()
        return False, msg
    finally:
        if list_file is not None and list_file.exists():
            try:
                list_file.unlink()
            except OSError:
                pass


def concat_videos_with_retry(
    video_paths: list[Path], output_path: Path, max_retries: int = 3
) -> tuple[bool, str]:
    """Retry wrapper around :func:`concat_videos`."""
    max_retries = max(1, int(max_retries))
    last_error = ""
    for attempt in range(1, max_retries + 1):
        ok, err = concat_videos(video_paths=video_paths, output_path=output_path)
        if ok:
            return True, err
        last_error = err
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        if attempt < max_retries:
            print(f"[WARN] 第{attempt}次concat失败，{attempt}s后重试…")
            time.sleep(attempt)

    print(f"[ERROR] concat重试失败(max={max_retries})")
    return False, last_error


# ── Concat: re-encode ─────────────────────────────────────────────────────────

def concat_videos_reencode(
    video_paths: list[Path],
    output_path: Path,
    crf: int = 18,
    audio_bitrate: str = "192k",
    target_resolution: str = "1920:1080",
    target_fps: int = 30,
    threads: int | None = None,
) -> tuple[bool, str]:
    """
    Concatenate videos via ``filter_complex`` with re-encoding.

    Normalises resolution, frame rate, pixel format and audio sample rate
    before joining, which avoids codec/time_base incompatibility issues.
    Returns ``(success, error_str)``.
    """
    if not video_paths:
        return False, "[ERROR] 未提供视频路径"

    normalized = [Path(p) for p in video_paths]
    missing = [str(p) for p in normalized if not p.exists()]
    if missing:
        return False, f"[ERROR] 文件不存在: {missing}"

    n = len(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".reencode.tmp" + output_path.suffix)

    try:
        if temp_path.exists():
            temp_path.unlink()

        v_filters = [
            f"[{i}:v]scale={target_resolution},fps={target_fps},"
            f"format=yuv420p,setsar=1[v{i}]"
            for i in range(n)
        ]
        a_filters = [
            f"[{i}:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]"
            for i in range(n)
        ]
        concat_in      = "".join(f"[v{i}][a{i}]" for i in range(n))
        concat_filter  = f"{concat_in}concat=n={n}:v=1:a=1[outv][outa]"
        filter_complex = ";".join(v_filters + a_filters + [concat_filter])

        cmd = ["ffmpeg", "-y"]
        for p in normalized:
            cmd += ["-i", str(p)]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
        ]
        if threads is not None:
            cmd += ["-threads", str(threads)]
        cmd.append(str(temp_path))

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        error_str = result.stderr.strip()
        if result.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"重编码失败(exit {result.returncode}): {error_str}"

        temp_path.replace(output_path)
        return True, error_str

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, f"重编码异常: {e}"
