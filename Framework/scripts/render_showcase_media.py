"""Render showcase screenshots and videos from the WebGL viewer (headless).

This script is intended for open-source-ready media generation:
- It starts a local HTTP server under the showcase directory.
- It uses Playwright (Chromium) to render the viewer.
- It exports PNG screenshots and MP4 clips.

Notes:
- Requires: `python -m pip install playwright` and `python -m playwright install chromium`
- MP4 encoding uses `imageio-ffmpeg` (already included in many Python stacks).
- Headless video capture is CPU-bound in many environments (SwiftShader). For best quality/smoothness,
  consider running the viewer on a GPU machine and recording with an external screen recorder.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import shutil
import subprocess
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import List, Tuple

import imageio_ffmpeg


def _start_http_server(root: Path, port: int = 0) -> Tuple[ThreadingHTTPServer, int]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    port = int(httpd.server_address[1])
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _encode_frames_to_mp4(
    frames_dir: Path,
    mp4: Path,
    *,
    fps: int,
    crf: int = 12,
    preset: str = "slow",
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-preset",
        str(preset),
        "-movflags",
        "+faststart",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def _convert_webm_to_mp4(webm: Path, mp4: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(webm),
        # Trim/cut is applied at the output stage for accuracy (no audio stream here).
        # If you see any leftover "loading" frames, increase the trim margin below.
        "-ss",
        "0",
        "-c:v",
        "libx264",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "12",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def _convert_webm_to_mp4_trimmed_scaled(
    webm: Path,
    mp4: Path,
    *,
    trim_start_s: float,
    duration_s: float,
    scale_w: int,
    scale_h: int,
    fps: int = 25,
    sharpen: float = 0.0,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"scale={int(scale_w)}:{int(scale_h)}:flags=lanczos"
    if float(sharpen) > 0:
        amt = float(sharpen)
        vf = f"{vf},unsharp=5:5:{amt:.3f}:5:5:0.000"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(webm),
        "-ss",
        f"{max(0.0, float(trim_start_s)):.3f}",
        "-t",
        f"{max(0.0, float(duration_s)):.3f}",
        "-r",
        str(int(fps)),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "12",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def _hstack_mp4(left: Path, right: Path, out: Path, *, crf: int = 12, preset: str = "slow") -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-preset",
        str(preset),
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


async def _capture_all(
    base_url: str,
    output_dir: Path,
    tmp_dir: Path,
    width: int,
    height: int,
    *,
    video_width: int,
    video_height: int,
    video_device_scale: float,
    fps: int,
    video_mode: str,
    chromium_gl: str,
    sharpen: float,
    headless: bool,
) -> None:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright is required. Install with: python -m pip install playwright && python -m playwright install chromium"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    screenshots: List[Tuple[str, str, float]] = [
        ("shot_01_overview_t2.png", "overview", 2.0),
        ("shot_02_overview_t14.png", "overview", 14.0),
        ("shot_03_chase_t8.png", "chase", 8.0),
        ("shot_04_chase_t22.png", "chase", 22.0),
        ("shot_05_orbit_t5.png", "orbit", 5.0),
        ("shot_06_orbit_t28.png", "orbit", 28.0),
    ]

    videos: List[Tuple[str, str, float, float]] = [
        # Split-screen like AirSim: (external view) | (UAV FPV)
        ("clip_01_overview.mp4", "overview", 0.0, 14.0),
        ("clip_02_chase.mp4", "chase", 12.0, 14.0),
        ("clip_03_orbit.mp4", "orbit", 24.0, 14.0),
    ]

    async with async_playwright() as p:
        launch_args = ["--disable-dev-shm-usage", "--no-sandbox"]
        if chromium_gl == "swiftshader":
            launch_args.insert(0, "--use-gl=swiftshader")
        browser = await p.chromium.launch(args=launch_args, headless=bool(headless))

        # Screenshots (no video recording needed)
        ctx = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
        page = await ctx.new_page()

        for fname, cam, t in screenshots:
            url = (
                f"{base_url}/viewer/index.html?camera={cam}"
                f"&t0={t}&speed=1"
                f"&ui=0&autoplay=0"
                f"&shadows=1&aa=1&detail=1&roads=0"
                f"&fog=0&wind=0&path=0&capture=1"
            )
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_function("window.__SHOWCASE_READY === true")
            # Give the renderer a few frames to settle
            await page.wait_for_timeout(250)
            await page.screenshot(path=str(output_dir / fname))

        await ctx.close()

        async def record_frames(cam: str, tag: str, *, t_start: float, dur: float, out_mp4: Path) -> None:
            frames_dir = tmp_dir / f"frames_{out_mp4.stem}_{tag}"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)

            target_half_w = int(width // 2)
            target_half_w -= target_half_w % 2
            target_h = int(height)
            target_h -= target_h % 2

            cap_w = int(video_width)
            cap_h = int(video_height)
            cap_w -= cap_w % 2
            cap_h -= cap_h % 2

            ctx = await browser.new_context(
                viewport={"width": int(cap_w), "height": int(cap_h)},
                device_scale_factor=float(video_device_scale),
            )
            page = await ctx.new_page()

            url = (
                f"{base_url}/viewer/index.html?camera={cam}"
                f"&t0={t_start}&speed=1&clarity=1"
                f"&ui=0&path=0&shadows=0&detail=0&roads=0"
                f"&fog=0&wind=0&autoplay=0&aa=1&fast=1&capture=1"
            )
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_function("window.__SHOWCASE_READY === true")

            # Pre-warm a stable frame at the start timestamp.
            await page.evaluate(
                """(t) => {
                  window.__SHOWCASE_SEEK(t);
                  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                }""",
                float(t_start),
            )

            frame_count = max(1, int(round(float(dur) * float(fps))))
            for i in range(frame_count):
                t = float(t_start) + (i / float(fps))
                await page.evaluate(
                    """(tt) => {
                      window.__SHOWCASE_SEEK(tt);
                      return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                    }""",
                    t,
                )
                await page.screenshot(path=str(frames_dir / f"frame_{i:05d}.png"))

            await page.close()
            await ctx.close()

            tmp_mp4 = tmp_dir / f"{out_mp4.stem}_{tag}_raw.mp4"
            _encode_frames_to_mp4(frames_dir, tmp_mp4, fps=int(fps), crf=12, preset="slow")
            _convert_webm_to_mp4_trimmed_scaled(
                tmp_mp4,
                out_mp4,
                trim_start_s=0.0,
                duration_s=float(dur),
                scale_w=target_half_w,
                scale_h=target_h,
                fps=int(fps),
                sharpen=float(sharpen),
            )
            with contextlib.suppress(Exception):
                tmp_mp4.unlink()
            shutil.rmtree(frames_dir, ignore_errors=True)

        async def record_view_mediarecorder(cam: str, tag: str, *, t_start: float, dur: float) -> Path:
            # Record directly from the canvas via MediaRecorder (WebM), then transcode to MP4.
            # This typically yields higher bitrate and better perceptual quality than Playwright's built-in recorder.
            cap_w = int(video_width)
            cap_h = int(video_height)
            cap_w -= cap_w % 2
            cap_h -= cap_h % 2

            ctx = await browser.new_context(
                viewport={"width": int(cap_w), "height": int(cap_h)},
                device_scale_factor=float(video_device_scale),
                accept_downloads=True,
            )
            page = await ctx.new_page()

            url = (
                f"{base_url}/viewer/index.html"
                f"?camera={cam}"
                f"&t0={t_start}&speed=1"
                f"&clarity=1&fast=1&capture=1"
                f"&ui=0&path=0&shadows=0&detail=0&roads=0"
                f"&fog=0&wind=0&autoplay=0&aa=1"
                f"&fixed_fps={int(fps)}&max_dt={max(0.001, 1.0/float(fps)):.4f}"
            )
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_function("window.__SHOWCASE_READY === true")
            await page.wait_for_timeout(200)

            # Prefer a higher bitrate to preserve detail after hstack and upscaling.
            bitrate = 32_000_000 if cap_w * cap_h >= 1280 * 720 else 18_000_000
            filename = f"{Path(cam).stem}_{tag}.webm"
            async with page.expect_download() as dl_info:
                await page.evaluate(
                    """(args) => window.__SHOWCASE_RECORD(args)""",
                    {
                        "filename": filename,
                        "durationS": float(dur),
                        "fps": int(fps),
                        "bitrate": int(bitrate),
                    },
                )
            download = await dl_info.value

            webm_out = tmp_dir / f"mr_{Path(cam).stem}_{tag}.webm"
            await download.save_as(str(webm_out))
            await page.close()
            await ctx.close()

            target_half_w = int(width // 2)
            target_half_w -= target_half_w % 2
            target_h = int(height)
            target_h -= target_h % 2
            mp4_out = tmp_dir / f"mr_{Path(cam).stem}_{tag}.mp4"
            _convert_webm_to_mp4_trimmed_scaled(
                webm_out,
                mp4_out,
                trim_start_s=0.0,
                duration_s=float(dur),
                scale_w=target_half_w,
                scale_h=target_h,
                fps=int(fps),
                sharpen=float(sharpen),
            )
            with contextlib.suppress(Exception):
                webm_out.unlink()
            return mp4_out

        # Videos
        for fname, external, t_start, dur in videos:
            stem = Path(fname).stem
            target_half_w = int(width // 2)
            target_half_w -= target_half_w % 2
            target_h = int(height)
            target_h -= target_h % 2

            def make_url(cam: str) -> str:
                return (
                    f"{base_url}/viewer/index.html"
                    f"?camera={cam}"
                    f"&t0={t_start}&speed=1"
                    f"&clarity=1&path=0&shadows=0"
                    f"&autoplay=0&ui=0&detail=0&roads=0"
                    f"&fog=0&wind=0"
                    f"&fixed_fps={int(fps)}&max_dt={max(0.001, 1.0/float(fps)):.4f}"
                    f"&aa=1&fast=1&capture=1"
                )

            async def record_view(cam: str, tag: str) -> Path:
                rec_dir = tmp_dir / f"rec_{stem}_{tag}"
                if rec_dir.exists():
                    shutil.rmtree(rec_dir)
                rec_dir.mkdir(parents=True, exist_ok=True)

                ctx = await browser.new_context(
                    viewport={"width": int(video_width), "height": int(video_height)},
                    device_scale_factor=float(video_device_scale),
                    record_video_dir=str(rec_dir),
                    record_video_size={"width": int(video_width), "height": int(video_height)},
                )
                page = await ctx.new_page()
                page_created = time.monotonic()

                await page.goto(make_url(cam), wait_until="networkidle")
                await page.wait_for_function("window.__SHOWCASE_READY === true")
                # Let the first frame present (avoid capturing any transient states)
                await page.wait_for_timeout(150)
                play_clicked = time.monotonic()
                await page.evaluate("document.getElementById('btnPlay').click()")
                await page.wait_for_timeout(int((dur + 0.8) * 1000))
                await page.close()

                video_path = await page.video.path()  # type: ignore[union-attr]
                await ctx.close()

                raw = Path(video_path)
                webm_out = tmp_dir / f"{stem}_{tag}.webm"
                shutil.move(str(raw), str(webm_out))
                mp4_out = tmp_dir / f"{stem}_{tag}.mp4"

                # Trim everything before play (plus a small margin to avoid any 1-frame UI flash).
                trim_s = max(0.8, play_clicked - page_created + 0.25)
                _convert_webm_to_mp4_trimmed_scaled(
                    webm_out,
                    mp4_out,
                    trim_start_s=trim_s,
                    duration_s=float(dur),
                    scale_w=target_half_w,
                    scale_h=target_h,
                    fps=int(fps),
                    sharpen=float(sharpen),
                )
                return mp4_out

            if video_mode == "frames":
                left_mp4 = tmp_dir / f"{stem}_ext.mp4"
                right_mp4 = tmp_dir / f"{stem}_fpv.mp4"
                await record_frames(external, "ext", t_start=t_start, dur=dur, out_mp4=left_mp4)
                await record_frames("fpv", "fpv", t_start=t_start, dur=dur, out_mp4=right_mp4)
            elif video_mode == "mediarecorder":
                left_mp4 = await record_view_mediarecorder(external, "ext", t_start=t_start, dur=dur)
                right_mp4 = await record_view_mediarecorder("fpv", "fpv", t_start=t_start, dur=dur)
            else:
                left_mp4 = await record_view(external, "ext")
                right_mp4 = await record_view("fpv", "fpv")

            final_out = output_dir / fname
            _hstack_mp4(left_mp4, right_mp4, final_out, crf=12, preset="slow")

            for p in (left_mp4, right_mp4):
                with contextlib.suppress(Exception):
                    p.unlink()

        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--showcase-dir", type=str, default="showcase", help="Path to showcase folder (contains viewer/ and inputs/)")
    parser.add_argument("--output-dir", type=str, default="showcase/media", help="Output directory for final media")
    parser.add_argument("--tmp-dir", type=str, default="showcase/tmp", help="Temporary directory")
    # Default to 4K outputs for showcase-grade clarity.
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (frames mode is deterministic)")
    parser.add_argument(
        "--video-mode",
        type=str,
        default="recorder",
        choices=["frames", "recorder", "mediarecorder"],
        help="Video pipeline: frames (offline, smooth), recorder (Playwright real-time), or mediarecorder (canvas captureStream)",
    )
    parser.add_argument(
        "--chromium-gl",
        type=str,
        default="auto",
        choices=["auto", "swiftshader"],
        help="Chromium GL backend: auto (prefer GPU if available) or swiftshader (CPU-only software GL)",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.25,
        help="Optional unsharp mask amount after upscaling (0 disables).",
    )
    parser.add_argument("--video-width", type=int, default=960, help="Internal per-panel video viewport width (will be upscaled)")
    parser.add_argument("--video-height", type=int, default=1080, help="Internal per-panel video viewport height (will be upscaled)")
    parser.add_argument(
        "--video-device-scale",
        type=float,
        default=1.0,
        help="Device scale factor for video contexts (2.0 often yields crisper 4K after hstack/upscale).",
    )
    parser.add_argument(
        "--headless",
        type=int,
        default=1,
        help="Run Chromium headless (1) or headed (0). Headed + GPU often yields the best quality.",
    )
    args = parser.parse_args()

    showcase_dir = Path(args.showcase_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()

    httpd, port = _start_http_server(showcase_dir, port=0)
    base_url = f"http://127.0.0.1:{port}"
    try:
        asyncio.run(
            _capture_all(
                base_url,
                output_dir,
                tmp_dir,
                int(args.width),
                int(args.height),
                video_width=int(args.video_width),
                video_height=int(args.video_height),
                video_device_scale=float(args.video_device_scale),
                fps=int(args.fps),
                video_mode=str(args.video_mode),
                chromium_gl=str(args.chromium_gl),
                sharpen=float(args.sharpen),
                headless=bool(int(args.headless)),
            )
        )
    finally:
        with contextlib.suppress(Exception):
            httpd.shutdown()

    print(f"Media written to: {output_dir}")


if __name__ == "__main__":
    main()
