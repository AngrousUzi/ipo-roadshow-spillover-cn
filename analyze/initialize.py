from pathlib import Path
import shutil
import subprocess
import time
import ffmpeg
import pandas as pd
import sys
import re
import tempfile
import multiprocessing
import os

# from polars import date
from config import get_audio_dir, get_video_dir, get_trans_dir, PROJECT_ROOT

VIDEO_OUTPUT_DIR = PROJECT_ROOT / "路演视频"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "路演音频"
TRANS_OUTPUT_DIR = PROJECT_ROOT / "路演转录"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARALLEL_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", "16"))

if os.name == "nt":  # Windows
    VIDEO_OPERATRION_PATH = PROJECT_ROOT / "videos"
    INDEX_PATH = PROJECT_ROOT / "anns" / "IPO_index_selected_platforms.xlsx"
    sys.path.insert(0, str(VIDEO_OPERATRION_PATH.resolve()))
    from audio_extract import extract_task
    # from audio_transcribe import transcribe_tasks
else:
    VIDEO_OPERATRION_PATH = PROJECT_ROOT / ".." / "roadshow-cn"
    DATA_ROOT = PROJECT_ROOT / ".."
    INDEX_PATH = DATA_ROOT / "IPO_index_selected_platforms.xlsx"
    sys.path.insert(0, str(VIDEO_OPERATRION_PATH.resolve()))
    # print(f"已添加视频处理模块路径: {VIDEO_OPERATRION_PATH}")
    from audio_extract import extract_task
    # from audio_transcribe import transcribe_tasks


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_video_integrity(video_path: Path) -> str:
    """
    运行 ffmpeg -v error -i <video> -c copy -f null - 验证视频完整性。
    返回 stderr 内容；无报错则返回空字符串。
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


def collect_audio_tasks() -> list[tuple[str, Path | None]]:
    """
    根据路演信息表，为每场路演生成最终处理用音频路径队列（每场路演 1 个音频）。

    返回
    ----
    list of (index2009 : str, audio_path : Path | None)
        每场路演对应一个元组；audio_path 为 None 表示文件缺失。
    """
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()

    results: list[tuple[str, Path | None]] = []

    for _, row in df_index.iterrows():
        platform  = str(row["采用视频平台"]).strip()
        index2009 = str(row.get("INDEX2009", "")).strip()
        code      = str(row.get(f"{platform}_去重代码", "")).strip()
        date      = str(row.get(f"{platform}_日期",     "")).strip()

        audio_path = AUDIO_OUTPUT_DIR / f"{index2009}_{code}_{date}.wav"
        if not audio_path.exists():
            print(
                f"[ERROR] 未找到音频文件: index={index2009} "
                f"code={code} date={date} @ {platform}"
            )
            results.append((index2009, None))
            continue

        results.append((index2009, audio_path))

    valid_count = sum(1 for _, p in results if p is not None)
    print(f"共生成 {valid_count} 个音频路径（总计 {len(results)} 场路演）。")
    return results


def clip_video(video_path: Path, output_path: Path) -> tuple[Path, str]:
    """
    对直接使用的视频复制到 output_path，并运行 ffmpeg 完整性检查。

    返回 (output_path, error_str)；error_str 为空字符串表示无报错。
    未来可在此按时间戳调用 ffmpeg 裁剪（例如去掉片头宣传或片尾内容），
    届时替换 shutil.copy2 为 ffmpeg 裁剪输出，不覆盖原始视频。
    """
    if output_path.exists():
        error_str = check_video_integrity(output_path)
        return output_path, error_str

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: 根据元数据（如 promo_timestamps）实现 ffmpeg 裁剪逻辑，替换下方复制
    shutil.copy2(video_path, output_path)
    error_str = check_video_integrity(output_path)
    return output_path, error_str


def concat_videos(video_paths: list[Path], output_path: Path) -> tuple[bool, str]:
    """使用 ffmpeg concat demuxer 拼接视频，返回 (success, error_str)。"""
    if not video_paths:
        msg = "[ERROR] 未提供可拼接的视频路径。"
        print(msg)
        return False, msg

    normalized_paths = [Path(p) for p in video_paths]
    missing = [str(p) for p in normalized_paths if not p.exists()]
    if missing:
        msg = f"[ERROR] 以下视频文件不存在，无法拼接: {missing}"
        print(msg)
        return False, msg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)

    # ffmpeg concat demuxer 需要一个列表文件，每行格式为: file 'path'
    list_file_path: Path | None = None
    error_str = ""
    try:
        if temp_output_path.exists():
            temp_output_path.unlink()

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for p in normalized_paths:
                escaped = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            list_file_path = Path(f.name)

        _, err = (
            ffmpeg
            .input(str(list_file_path), format="concat", safe=0)
            .output(str(temp_output_path), c="copy")
            .global_args("-loglevel", "error")
            .overwrite_output()
            .run(capture_stderr=True)
        )
        if err:
            err_decoded = err.decode("utf8", errors="ignore").strip()
            if err_decoded:
                print(f"{output_path}: FFmpeg stderr: {err_decoded}")
                error_str = err_decoded

        temp_output_path.replace(output_path)
        return True, error_str
    except ffmpeg.Error as e:
        lines = ["FFmpeg 执行失败:"]
        if e.stderr:
            lines.append(e.stderr.decode("utf8", errors="ignore"))
        msg = "\n".join(lines)
        print(msg)
        if temp_output_path.exists():
            temp_output_path.unlink()
        return False, msg
    except Exception as e:
        msg = f"发生错误: {e}"
        print(msg)
        if temp_output_path.exists():
            temp_output_path.unlink()
        return False, msg
    finally:
        if list_file_path is not None and list_file_path.exists():
            try:
                list_file_path.unlink()
            except OSError:
                pass


def concat_videos_with_retry(
    video_paths: list[Path], output_path: Path, max_retries: int = 3
) -> tuple[bool, str]:
    """拼接失败时自动重试，返回 (success, last_error_str)；全部失败返回 (False, error)。"""
    max_retries = max(1, int(max_retries))
    last_error = ""
    for attempt in range(1, max_retries + 1):
        success, error_str = concat_videos(video_paths=video_paths, output_path=output_path)
        if success:
            return True, error_str

        last_error = error_str
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

        if attempt < max_retries:
            wait_seconds = attempt
            print(f"[WARN] 第 {attempt} 次拼接失败，{wait_seconds} 秒后重试...")
            time.sleep(wait_seconds)

    print(f"[ERROR] 视频拼接重试失败，已达到最大次数: {max_retries}")
    return False, last_error


def concat_videos_reencode(
    video_paths: list[Path],
    output_path: Path,
    crf: int = 18,
    audio_bitrate: str = "192k",
    target_resolution: str = "1920:1080",
    target_fps: int = 30,
) -> tuple[bool, str]:
    """
    使用 ffmpeg filter_complex concat 重编码拼接视频。

    统一缩放分辨率、帧率、音频采样率后再拼接，彻底规避编解码器类型
    （H.264 vs HEVC）、time_base、SPS/PPS、profile/level 不一致引起的问题。

    返回 (success, error_str)。
    """
    if not video_paths:
        return False, "[ERROR] 未提供视频路径"

    normalized = [Path(p) for p in video_paths]
    missing = [str(p) for p in normalized if not p.exists()]
    if missing:
        return False, f"[ERROR] 视频文件不存在: {missing}"

    n = len(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".reencode.tmp" + output_path.suffix)

    try:
        if temp_path.exists():
            temp_path.unlink()

        # 每路视频流：统一分辨率、帧率、像素格式；音频：统一到立体声 44100Hz
        v_filters = [
            f"[{i}:v]scale={target_resolution},fps={target_fps},"
            f"format=yuv420p,setsar=1[v{i}]"
            for i in range(n)
        ]
        a_filters = [
            f"[{i}:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]"
            for i in range(n)
        ]
        concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        concat_filter = f"{concat_in}concat=n={n}:v=1:a=1[outv][outa]"
        filter_complex = ";".join(v_filters + a_filters + [concat_filter])

        cmd = ["ffmpeg", "-y"]
        for p in normalized:
            cmd += ["-i", str(p)]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            str(temp_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        error_str = result.stderr.strip()

        if result.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"重编码失败 (exit {result.returncode}): {error_str}"

        temp_path.replace(output_path)
        return True, error_str

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, f"重编码异常: {e}"


# ── Multiprocessing worker ─────────────────────────────────────────────────────

def process_video_row(row_data: dict) -> tuple[str, str | None, bool, str]:
    """
    处理单行路演数据，返回 (index2009, video_path_str | None, success, error_str)。
    作为 multiprocessing.Pool worker 使用（须为 module-level 函数）。
    """
    platform     = row_data["platform"]
    index2009    = row_data["index2009"]
    code         = row_data["code"]
    date         = row_data["date"]
    video_number = row_data["video_number"]
    video_dir    = Path(row_data["video_dir"])
    output_path  = Path(row_data["output_path"])

    if not video_dir.exists():
        print(f"[WARN] 视频目录不存在: {video_dir}")
        return index2009, None, False, "视频目录不存在"

    # ── 上证 / 中国证券网：取视频1、2拼接 ──────────────────────────────────
    if platform in ("上证", "中国证券网"):
        if output_path.exists():
            err = check_video_integrity(output_path)
            if err == "":
                return index2009, str(output_path), True, err
            print(f"[WARN] 视频完整性检查报错: {output_path}")
            try:
                output_path.unlink()
            except OSError:
                pass

        seg_map: dict[int, Path] = {}
        for vf in video_dir.glob(f"{code}_*_{date}_视频*.mp4"):
            m = re.search(r"_视频(\d+)", vf.name)
            if m:
                num_match = m.group(1)
                if num_match.isdigit() and num_match in ("1", "2"):
                    seg_map[int(num_match)] = vf

        video_paths = [seg_map[k] for k in sorted(seg_map)]
        expected_count = min(2, video_number)
        if len(video_paths) != expected_count:
            print(
                f"[WARN] 预期 {expected_count} 段，实际找到 {len(video_paths)} 段 "
                f"(index={index2009}, code={code}, date={date}, platform={platform})"
            )

        if not video_paths:
            msg = f"未找到任何视频段: index={index2009} code={code} date={date}"
            print(f"[ERROR] {msg}")
            return index2009, None, False, msg

        success, error_str = concat_videos_with_retry(video_paths, output_path)
        if success:
            integrity_err = check_video_integrity(output_path)
            if not integrity_err:
                return index2009, str(output_path), True, error_str
            print(f"[WARN] copy 拼接完整性异常，改用重编码: {output_path}")
            try:
                output_path.unlink()
            except OSError:
                pass
            error_str = integrity_err
        else:
            print(f"[WARN] 视频 copy 拼接失败，尝试重编码 (crf=18, aac 192k)... ({output_path})")
        success, error_str = concat_videos_reencode(video_paths, output_path)
        if success:
            return index2009, str(output_path), True, error_str
        return index2009, None, False, error_str

    # ── 中证（多段）──────────────────────────────────────────────────────────
    elif platform == "中证" and video_number > 1:
        if output_path.exists():
            err = check_video_integrity(output_path)
            if err == "":
                return index2009, str(output_path), True, err
            print(f"[WARN] 视频完整性检查报错: {output_path}")
            try:
                output_path.unlink()
            except OSError:
                pass

        seg_map: dict[int, Path] = {}
        for vf in video_dir.glob(f"{code}_*_{date}_视频*.mp4"):
            m = re.search(r"_视频(\d+)", vf.name)
            if m:
                num_match = m.group(1)
                # 001217 华尔泰:3->2->1  301000 肇民科技 etc.
                if (
                    code in ("001217", "301000", "301022", "301093", "301149", "301180", "301230")
                    and num_match.isdigit()
                    and num_match in ("1", "2", "3")
                ):
                    seg_map[int(num_match)] = vf
                # 301161 唯万密封 2->1
                elif code == "301161" and num_match.isdigit() and num_match in ("1", "2"):
                    seg_map[int(num_match)] = vf

        video_paths = [seg_map[k] for k in sorted(seg_map, reverse=True)]
        expected_count = (
            3 if code in ("001217", "301000", "301022", "301093", "301149", "301180", "301230")
            else 2
        )
        if len(video_paths) != expected_count:
            print(
                f"[WARN] 预期 {expected_count} 段，实际找到 {len(video_paths)} 段 "
                f"(index={index2009}, code={code}, date={date}, platform={platform})"
            )

        if not video_paths:
            msg = f"未找到任何视频段: index={index2009} code={code} date={date}"
            print(f"[ERROR] {msg}")
            return index2009, None, False, msg

        success, error_str = concat_videos_with_retry(video_paths, output_path)
        if success:
            integrity_err = check_video_integrity(output_path)
            if not integrity_err:
                return index2009, str(output_path), True, error_str
            print(f"[WARN] copy 拼接完整性异常，改用重编码: {output_path}")
            try:
                output_path.unlink()
            except OSError:
                pass
            error_str = integrity_err
        else:
            print(f"[WARN] 视频 copy 拼接失败，尝试重编码 (crf=18, aac 192k)... ({output_path})")
        success, error_str = concat_videos_reencode(video_paths, output_path)
        if success:
            return index2009, str(output_path), True, error_str
        return index2009, None, False, error_str

    # ── 直接使用：中证（单段）/ 全景 / IR ────────────────────────────────────
    else:
        candidates = [
            vf for vf in video_dir.glob(f"{code}_*_{date}*.mp4")
            if "宣传片" not in vf.name
        ]
        if not candidates:
            msg = f"未找到视频: index={index2009} code={code} date={date} @ {platform}"
            print(f"[ERROR] {msg}")
            return index2009, None, False, msg

        if len(candidates) > 1:
            print(
                f"[WARN] 找到 {len(candidates)} 个候选视频，使用第 1 个: "
                f"{[vf.name for vf in candidates]}"
            )

        final_path, error_str = clip_video(candidates[0], output_path)
        return index2009, str(final_path), True, error_str


def collect_video_tasks() -> list[tuple[str, Path | None, bool, str]]:
    """
    根据路演信息表，为每场路演生成最终处理用视频路径队列（每场路演 1 个视频）。
    使用多进程并行处理拼接/复制操作。

    拼接规则（只保留路演回放，自动排除宣传片等非回放内容）：
    - 上证 / 中国证券网：将第1和第2个"_视频N"片段按序号升序拼接为 1 个文件
    - 中证（视频数量 > 1）：将所有非宣传片视频段按序号降序拼接为 1 个文件

    直接使用（经 clip_video 保留裁剪接口）：
    - 中证（视频数量 = 1）/ 全景 / IR：取匹配的单段视频直接使用

    Returns
    -------
    list of (index2009, video_path | None, success, error_str)
    """
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()

    row_data_list: list[dict] = []
    for _, row in df_index.iterrows():
        platform     = str(row["采用视频平台"]).strip()
        index2009    = str(row.get("INDEX2009", "")).strip()
        code         = str(row.get(f"{platform}_去重代码", "")).strip()
        date         = str(row.get(f"{platform}_日期",     "")).strip()
        video_number = int(str(row.get(f"{platform}_视频数量", "1")).strip())
        video_dir    = get_video_dir(platform)
        output_path  = VIDEO_OUTPUT_DIR / f"{index2009}_{code}_{date}.mp4"

        row_data_list.append({
            "platform":     platform,
            "index2009":    index2009,
            "code":         code,
            "date":         date,
            "video_number": video_number,
            "video_dir":    str(video_dir),
            "output_path":  str(output_path),
        })

    num_workers = max(1, PARALLEL_PROCESSES)
    with multiprocessing.Pool(processes=num_workers) as pool:
        raw_results = pool.map(process_video_row, row_data_list)

    results: list[tuple[str, Path | None, bool, str]] = [
        (idx, Path(vp) if vp is not None else None, ok, err)
        for idx, vp, ok, err in raw_results
    ]

    valid_count = sum(1 for _, p, _, _ in results if p is not None)
    print(f"共生成 {valid_count} 个路演视频路径（总计 {len(results)} 场路演）。")
    return results


if __name__ == "__main__":
    video_tasks = collect_video_tasks()

    # 将视频任务结果保存为 CSV（含转换状态与报错信息）
    df_video = pd.DataFrame([
        {
            "index2009":     idx,
            "video_path":    str(vp) if vp is not None else "",
            "success":       ok,
            "ffmpeg_errors": err,
        }
        for idx, vp, ok, err in video_tasks
    ])
    video_csv = PROJECT_ROOT / "video_tasks.csv"
    df_video.to_csv(video_csv, encoding="utf-8-sig", index=False)
    print(f"视频任务日志已保存: {video_csv}")

    # 将 video 序列提取音频
    for idx, vp, ok, _ in video_tasks:
        if vp is not None:
            output_audio_path = AUDIO_OUTPUT_DIR / (vp.stem + ".wav")
            if not output_audio_path.exists():
                extract_task((vp, output_audio_path))
            else:
                print(f"音频已存在，跳过提取: {output_audio_path}")
        else:
            print(f"[WARN] 路演 {idx} 视频路径缺失，无法提取音频。")

    audio_tasks = collect_audio_tasks()
    df_audio = pd.DataFrame([
        {
            "index2009":  idx,
            "audio_path": str(ap) if ap is not None else "",
        }
        for idx, ap in audio_tasks
    ])
    audio_csv = PROJECT_ROOT / "audio_tasks.csv"
    df_audio.to_csv(audio_csv, encoding="utf-8-sig", index=False)
    print(f"音频任务日志已保存: {audio_csv}")
