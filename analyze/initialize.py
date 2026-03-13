from pathlib import Path
import shutil
import time
import ffmpeg
import pandas as pd
import sys
import re
import tempfile

# from polars import date
from config import get_audio_dir, get_video_dir, get_trans_dir, PROJECT_ROOT
import os
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "路演视频"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "路演音频"
TRANS_OUTPUT_DIR = PROJECT_ROOT / "路演转录"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if os.name == "nt":
    VIDEO_OPERATRION_PATH = PROJECT_ROOT / "videos"
    INDEX_PATH = PROJECT_ROOT / "anns" / "IPO_index_selected_platforms.xlsx"
    sys.path.insert(0, str(VIDEO_OPERATRION_PATH.resolve()))
    from audio_extract import extract_task
    from audio_transcribe import transcribe_tasks
else:
    VIDEO_OPERATRION_PATH = PROJECT_ROOT / ".." / "roadshow-cn"
    DATA_ROOT = PROJECT_ROOT/ ".." 
    INDEX_PATH = DATA_ROOT / "IPO_index_selected_platforms.xlsx"
    sys.path.insert(0, str(VIDEO_OPERATRION_PATH.resolve()))
    # print(f"已添加视频处理模块路径: {VIDEO_OPERATRION_PATH}")
    from audio_extract import extract_task
    # from audio_transcribe import transcribe_tasks



def collect_video_tasks() -> list[tuple[str, Path | None]]:
    """
    根据路演信息表，为每场路演生成最终处理用视频路径队列（每场路演 1 个视频）。

    拼接规则（只保留路演回放，自动排除宣传片等非回放内容）：
    - 上证 / 中国证券网：将第1和第2个"_视频N"片段按序号升序拼接为 1 个文件
    - 中证（视频数量 > 1）：将所有非宣传片视频段按序号降序拼接为 1 个文件

    直接使用（经 _clip_video_for_analysis 保留裁剪接口）：
    - 中证（视频数量 = 1）/ 全景 / IR：取匹配的单段视频直接使用

    拼接输出命名：
      {platform}路演视频/{index2009}_{code}_{date}.mp4

    Returns
    -------
    list of (index2009 : str, video_path : Path | None)
        每场路演对应一个元组；video_path 为 None 表示文件缺失或处理失败。
    """
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()

    results: list[tuple[str, Path | None]] = []

    for _, row in df_index.iterrows():
        platform     = str(row["采用视频平台"]).strip()
        index2009    = str(row.get("INDEX2009", "")).strip()
        code         = str(row.get(f"{platform}_去重代码", "")).strip()
        date         = str(row.get(f"{platform}_日期",     "")).strip()
        video_number = int(str(row.get(f"{platform}_视频数量", "1")).strip())

        video_dir = get_video_dir(platform)
        if not video_dir.exists():
            print(f"[WARN] 视频目录不存在: {video_dir}")
            results.append((index2009, None))
            continue

        if platform in ("上证", "中国证券网"):
            output_path = VIDEO_OUTPUT_DIR / f"{index2009}_{code}_{date}.mp4"
            if output_path.exists():
                results.append((index2009, output_path))
                continue

            # 收集"视频N"片段，按序号升序排列，只取前 video_number 段
            # 宣传片通常编号最大（> video_number），按此截断可天然排除
            seg_map: dict[int, Path] = {}
            for vf in video_dir.glob(f"{code}_*_{date}_视频*.mp4"):
                m = re.search(r"_视频(\d+)", vf.name)
                if m:
                    num_match = m.group(1)
                    if num_match.isdigit() and num_match =="1" or num_match =="2":
                        seg_map[int(num_match)] = vf

            video_paths = [seg_map[k] for k in seg_map]
            
            expected_count = min(2, video_number)
            if len(video_paths) != expected_count:
                print(
                    f"[WARN] 预期 {expected_count} 段，实际找到 {len(video_paths)} 段 "
                    f"(index={index2009}, code={code}, date={date}, platform={platform})"
                )

            if not video_paths:
                print(f"[ERROR] 未找到任何视频段: index={index2009} code={code} date={date}")
                results.append((index2009, None))
                continue

            if concat_videos_with_retry(video_paths, output_path):
                results.append((index2009, output_path))
            else:
                results.append((index2009, None))
        elif platform == "中证" and video_number > 1:
            output_path = VIDEO_OUTPUT_DIR / f"{index2009}_{code}_{date}.mp4"
            if output_path.exists():
                results.append((index2009, output_path))
                continue

            seg_map: dict[int, Path] = {}
            for vf in video_dir.glob(f"{code}_*_{date}_视频*.mp4"):
                m = re.search(r"_视频(\d+)", vf.name)
                if m:
                    num_match = m.group(1)
                    if code in ("001217", "301000", "301022", "301093", "301149", "301180", "301230") and num_match.isdigit() and num_match in ("1", "2", "3"):
                        seg_map[int(num_match)] = vf
                    elif code =="301161" and num_match.isdigit() and num_match in ("1", "2"):
                        seg_map[int(num_match)] = vf
            # 001217 华尔泰:3->2->1
            # 301000 肇民科技 3->2->1
            # 301022 海泰科 3->2->1
            # 301093 华兰股份 3->2->1
            # 301149 隆华新材 3->2->1
            # 301180 万祥科技 3->2->1
            # 301230 泓博医药 3->2->1 
            # 301161 唯万密封 2->1
            video_paths = [seg_map[k] for k in sorted(seg_map,reverse=True)]

            expected_count =3 if code in ("001217", "301000", "301022", "301093", "301149", "301180", "301230") else 2
            if len(video_paths)!=3 and code in ("001217", "301000", "301022", "301093", "301149", "301180", "301230") or len(video_paths)!=2 and code =="301161":
                print(
                    f"[WARN] 预期 {expected_count} 段，实际找到 {len(video_paths)} 段 "
                    f"(index={index2009}, code={code}, date={date}, platform={platform})"
                )

            if not video_paths:
                print(f"[ERROR] 未找到任何视频段: index={index2009} code={code} date={date}")
                results.append((index2009, None))
                continue

            if concat_videos_with_retry(video_paths, output_path):
                results.append((index2009, output_path))
            else:
                results.append((index2009, None))

        # ── 直接使用：中证（单段）/ 全景 / IR ───────────────────────────────
        else:
            candidates = [
                vf for vf in video_dir.glob(f"{code}_*_{date}*.mp4")
                if "宣传片" not in vf.name
            ]
            if not candidates:
                print(
                    f"[ERROR] 未找到视频: index={index2009} "
                    f"code={code} date={date} @ {platform}"
                )
                results.append((index2009, None))
                continue

            if len(candidates) > 1:
                print(
                    f"[WARN] 找到 {len(candidates)} 个候选视频，使用第 1 个: "
                    f"{[vf.name for vf in candidates]}"
                )

            output_path = VIDEO_OUTPUT_DIR / f"{index2009}_{code}_{date}.mp4"
            final_path = clip_video(candidates[0], output_path)
            results.append((index2009, final_path))

    valid_count = sum(1 for _, p in results if p is not None)
    print(f"共生成 {valid_count} 个路演视频路径（总计 {len(results)} 场路演）。")
    return results

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

        audio_dir = get_audio_dir(platform)
        if not audio_dir.exists():
            print(f"[WARN] 音频目录不存在: {audio_dir}")
            results.append((index2009, None))
            continue

        audio_path = audio_dir / f"{code}_{date}.wav"
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

def clip_video(video_path: Path, output_path: Path) -> Path:
    """
    对直接使用的视频进行裁剪并输出到 output_path。
    当前暂未实现裁剪逻辑，直接复制原始文件到 output_path。
    未来可在此按时间戳调用 ffmpeg 裁剪（例如去掉片头宣传或片尾内容），
    届时替换 shutil.copy2 为 ffmpeg 裁剪输出，不覆盖原始视频。
    """
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: 根据元数据（如 promo_timestamps）实现 ffmpeg 裁剪逻辑，替换下方复制
    shutil.copy2(video_path, output_path)
    return output_path

def concat_videos(video_paths: list[Path], output_path: Path):
    """使用 ffmpeg concat demuxer 拼接视频，成功返回 True。"""
    if not video_paths:
        print("[ERROR] 未提供可拼接的视频路径。")
        return False

    normalized_paths = [Path(p) for p in video_paths]
    missing = [str(p) for p in normalized_paths if not p.exists()]
    if missing:
        print(f"[ERROR] 以下视频文件不存在，无法拼接: {missing}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)

    # ffmpeg concat demuxer 需要一个列表文件，每行格式为: file 'path'
    list_file_path: Path | None = None
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
            err_str = err.decode("utf8", errors="ignore")
            if err_str.strip():
                print(f"{output_path}: FFmpeg stderr: {err_str}")

        temp_output_path.rename(output_path)
        return True
    except ffmpeg.Error as e:
        print("FFmpeg 执行失败:")
        if e.stderr:
            print(e.stderr.decode("utf8", errors="ignore"))
        if temp_output_path.exists():
            temp_output_path.unlink()
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
            return False
    finally:
        if list_file_path is not None and list_file_path.exists():
            try:
                list_file_path.unlink()
            except OSError:
                pass

def concat_videos_with_retry(video_paths: list[Path], output_path: Path, max_retries: int = 3):
    """拼接失败时自动重试，全部失败返回 False。"""
    max_retries = max(1, int(max_retries))
    for attempt in range(1, max_retries + 1):
        if concat_videos(video_paths=video_paths, output_path=output_path):
            return True

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
    return False

if __name__ == "__main__":
    video_tasks = collect_video_tasks()
    # 将video序列提取音频
    with open (PROJECT_ROOT/"videos.txt", "wb") as f:
        for video_task in video_tasks:
            index2009, video_path = video_task
            f.write(f"{index2009}\t{video_path}\n".encode("utf-8"))
    audio_tasks_to_extract= [AUDIO_OUTPUT_DIR / (video_file.stem + ".wav") if video_file is not None else None for _, video_file in video_tasks]
    for video_task, audio_task in zip(video_tasks, audio_tasks_to_extract):
        index2009, video_path = video_task
        if video_path is not None:
            output_audio_path = AUDIO_OUTPUT_DIR / (video_path.stem + ".wav")
            if not output_audio_path.exists():
                extract_task((video_path, output_audio_path))
            else:
                print(f"音频已存在，跳过提取: {output_audio_path}")
        else:
            print(f"[WARN] 路演 {index2009} 视频路径缺失，无法提取音频。")
            
    audio_tasks = collect_audio_tasks()
    with open (PROJECT_ROOT/"audios.txt", "wb") as f:
        for audio_task in audio_tasks:
            index2009, audio_path = audio_task
            f.write(f"{index2009}\t{audio_path}\n".encode("utf-8"))
