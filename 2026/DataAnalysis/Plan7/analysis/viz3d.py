"""
3D/動画可視化ユーティリティ（Ver2基盤）

本モジュールは、WIPの時間×ノードの推移を3Dサーフェス/アニメーションで出力する最小実装です。
- 依存: matplotlib（PillowWriterによるGIF出力をデフォルト）、ffmpegがあればmp4も可。
- 入力: engine.results["wip_history"] 互換のリスト[{time: float, node_wip: {node_id: int}, ...}]
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI環境に依存しないバックエンド
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 必要なインポート（副作用）


def _prepare_wip_mesh(wip_history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """wip_historyから3Dサーフェス用の(X:time, Y:node_index, Z:wip)メッシュを生成する。

    Returns:
        X (T×N), Y (T×N), Z (T×N), nodes(list of str), times(1D)
    """
    if not wip_history:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), ["EMPTY"], np.array([0.0])

    # 時系列でソート
    wip_sorted = sorted(wip_history, key=lambda r: float(r.get("time", 0.0)))
    times = np.array([float(r.get("time", 0.0)) for r in wip_sorted], dtype=float)

    # すべてのノードIDの集合
    node_ids = set()
    for r in wip_sorted:
        node_ids.update((r.get("node_wip") or {}).keys())
    nodes = sorted(node_ids)
    n_nodes = max(1, len(nodes))
    n_times = len(times)

    # Z行列の構築（不足は0で埋める）
    Z = np.zeros((n_times, n_nodes), dtype=float)
    node_index = {nid: i for i, nid in enumerate(nodes)}
    for t_idx, r in enumerate(wip_sorted):
        nw = r.get("node_wip") or {}
        for nid, v in nw.items():
            Z[t_idx, node_index[nid]] = float(v or 0)

    # メッシュ
    X = np.tile(times.reshape(-1, 1), (1, n_nodes))
    Y = np.tile(np.arange(n_nodes).reshape(1, -1), (n_times, 1))
    return X, Y, Z, nodes, times


# ---------------- スタイルプリセットと適用ユーティリティ ----------------

# 可視化ごとのスタイルプリセット
_STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    # 共通: cmap, bg_color, fg_color, grid, alpha
    # surface系: wireframe(bool), edgecolor, linewidth, antialiased, colorbar
    # scatter/line系: marker, markersize, palette(list[str])
    # network系: node_color, node_size, node_edgecolor, edge_color, edge_alpha, text_color, axis_off
    "Default": {
        "cmap": "viridis",
        "alpha": 1.0,
        "edgecolor": "none",
        "antialiased": True,
        "colorbar": True,
        "bg_color": "white",
        "fg_color": "black",
        "grid": False,
        # network defaults
        "node_color": "steelblue",
        "node_size": 80,
        "node_edgecolor": None,
        "edge_color": "gray",
        "edge_alpha": 0.6,
        "text_color": None,
        "axis_off": True,
    },
    "Dark": {
        "cmap": "viridis",
        "alpha": 1.0,
        "edgecolor": "none",
        "antialiased": True,
        "colorbar": True,
        "bg_color": "#222222",
        "fg_color": "#eeeeee",
        "grid": False,
        "node_color": "#4aa3ff",
        "node_size": 90,
        "node_edgecolor": "#0a0a0a",
        "edge_color": "#bbbbbb",
        "edge_alpha": 0.7,
        "text_color": "#f0f0f0",
        "axis_off": True,
    },
    "Publication": {
        "cmap": "cividis",
        "alpha": 0.95,
        "edgecolor": "#333333",
        "linewidth": 0.2,
        "antialiased": True,
        "colorbar": True,
        "bg_color": "white",
        "fg_color": "black",
        "grid": False,
        "node_color": "#1f77b4",
        "node_size": 70,
        "node_edgecolor": "#1a1a1a",
        "edge_color": "#4d4d4d",
        "edge_alpha": 0.6,
        "text_color": "#1a1a1a",
        "axis_off": False,
    },
    "Wireframe": {
        "wireframe": True,
        "linewidth": 0.6,
        "edgecolor": "#666666",
        "alpha": 1.0,
        "antialiased": True,
        "colorbar": False,
        "bg_color": "white",
        "fg_color": "black",
        "grid": True,
        "node_color": "#2ca02c",
        "node_size": 70,
        "edge_color": "#666666",
        "edge_alpha": 0.6,
        "axis_off": False,
    },
    "HighContrast": {
        "cmap": "plasma",
        "alpha": 1.0,
        "edgecolor": "none",
        "antialiased": True,
        "colorbar": True,
        "bg_color": "white",
        "fg_color": "black",
        "grid": False,
        "node_color": "#d62728",
        "node_size": 85,
        "edge_color": "#000000",
        "edge_alpha": 0.7,
        "text_color": "#000000",
        "axis_off": False,
    },
    "Monochrome": {
        "cmap": "Greys",
        "alpha": 1.0,
        "edgecolor": "none",
        "antialiased": True,
        "colorbar": True,
        "bg_color": "white",
        "fg_color": "black",
        "grid": False,
        "node_color": "#4d4d4d",
        "node_size": 75,
        "edge_color": "#7f7f7f",
        "edge_alpha": 0.6,
        "text_color": "#1a1a1a",
        "axis_off": True,
    },
}


def _resolve_style(style: Any | None) -> Dict[str, Any]:
    if style is None:
        return dict(_STYLE_PRESETS["Default"])  # デフォルト
    if isinstance(style, str):
        return dict(_STYLE_PRESETS.get(style, _STYLE_PRESETS["Default"]))
    if isinstance(style, dict):
        base = dict(_STYLE_PRESETS["Default"])  # 既定をベースに上書き
        base.update(style)
        return base
    return dict(_STYLE_PRESETS["Default"])  # フォールバック


def _apply_common(ax, fig, style: Dict[str, Any]):
    # 背景・前景
    bg = style.get("bg_color")
    if bg is not None:
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
    fg = style.get("fg_color")
    if fg is not None:
        for spine in getattr(ax, 'spines', {}).values() if hasattr(ax, 'spines') else []:
            spine.set_color(fg)
        for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick.set_color(fg)
        # 軸ラベル/タイトル
        if ax.get_title():
            ax.title.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.zaxis.label.set_color(fg)
    # グリッド
    if style.get("grid"):
        ax.grid(True, color=style.get("grid_color", "#cccccc"), alpha=0.7)
    else:
        ax.grid(False)


def save_wip_surface_png(
    wip_history: List[Dict[str, Any]],
    outfile: str,
    title: str = "WIP Surface (3D)",
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """WIPの3Dサーフェス静止画を保存する。戻り値は出力パス。"""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    X, Y, Z, nodes, _ = _prepare_wip_mesh(wip_history)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)
    _apply_common(ax, fig, st)
    # サーフェス描画
    if st.get("wireframe"):
        surf = ax.plot_wireframe(
            X, Y, Z,
            rstride=st.get("rstride", 1), cstride=st.get("cstride", 1),
            linewidth=st.get("linewidth", 0.6), color=st.get("edgecolor", "#666666"),
        )
    else:
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=st.get("cmap", "viridis"), edgecolor=st.get("edgecolor", "none"),
            antialiased=bool(st.get("antialiased", True)), alpha=st.get("alpha", 1.0),
        )
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Node')
    ax.set_zlabel('WIP')
    # y軸の目盛をノード名に
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes, rotation=0, fontsize=7)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else 25, azim=azim if azim is not None else 45)
    if st.get("colorbar", True) and not st.get("wireframe"):
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    fig.savefig(outfile, dpi=int(st.get("dpi", 120)))
    plt.close(fig)
    return outfile


def render_wip_surface_video(
    wip_history: List[Dict[str, Any]],
    outfile: str,
    fps: int = 10,
    rotate: bool = True,
    frame_step: int = 1,
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """
    WIP 3Dサーフェスのアニメーションを生成する。
    - デフォルトはPillowWriterでGIF出力（拡張子が.mp4の場合、ffmpegが見つかればmp4、それ以外は.gifにフォールバック）
    - rotate=True の場合、時間切替に加えて視点をゆっくり回転
    戻り値は実際に保存したファイルパス。
    """
    # 出力拡張子の決定とフォールバック
    base_dir = os.path.dirname(outfile) or "."
    os.makedirs(base_dir, exist_ok=True)
    root, ext = os.path.splitext(outfile)
    ext = ext.lower()

    X, Y, Z, nodes, times = _prepare_wip_mesh(wip_history)
    # フレーム間引き（frame_step）
    try:
        step = max(1, int(frame_step or 1))
    except Exception:
        step = 1
    if step > 1:
        idx = np.arange(0, Z.shape[0], step, dtype=int)
        if len(idx) == 0 or idx[-1] != Z.shape[0] - 1:
            # 最終フレームは必ず含める
            idx = np.append(idx, Z.shape[0] - 1)
        Z = Z[idx, :]
        times = times[idx]
        # X/Yを再生成
        Tdec, N = Z.shape
        X = np.tile(times.reshape(-1, 1), (1, N))
        Y = np.tile(np.arange(N).reshape(1, -1), (Tdec, 1))
    T, N = Z.shape
    if T == 0 or N == 0:
        # 空であれば静止画のみ
        return save_wip_surface_png(wip_history, root + "_empty.png")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)

    # 初期フレーム
    if st.get("wireframe"):
        surf = [ax.plot_wireframe(
            X[:1, :], Y[:1, :], Z[:1, :],
            rstride=st.get("rstride", 1), cstride=st.get("cstride", 1),
            linewidth=st.get("linewidth", 0.6), color=st.get("edgecolor", "#666666"),
        )]
    else:
        surf = [ax.plot_surface(
            X[:1, :], Y[:1, :], Z[:1, :],
            cmap=st.get("cmap", "viridis"), edgecolor=st.get("edgecolor", "none"),
            antialiased=bool(st.get("antialiased", True)), alpha=st.get("alpha", 1.0),
        )]
    ax.set_xlabel('Time')
    ax.set_ylabel('Node')
    ax.set_zlabel('WIP')
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes, rotation=0, fontsize=7)

    # Zの最大値に合わせてスケール固定
    zmax = max(1.0, float(np.max(Z)))
    ax.set_zlim(0, zmax)

    base_elev = 25 if elev is None else float(elev)
    base_azim = 0 if azim is None else float(azim)

    def init():
        ax.clear()
        _apply_common(ax, fig, st)
        ax.set_xlabel('Time')
        ax.set_ylabel('Node')
        ax.set_zlabel('WIP')
        ax.set_yticks(np.arange(len(nodes)))
        ax.set_yticklabels(nodes, rotation=0, fontsize=7)
        ax.set_zlim(0, zmax)
        # 初期視点
        ax.view_init(elev=base_elev, azim=base_azim)
        return []

    def update(frame: int):
        ax.clear()
        ax.set_xlabel('Time')
        ax.set_ylabel('Node')
        ax.set_zlabel('WIP')
        ax.set_yticks(np.arange(len(nodes)))
        ax.set_yticklabels(nodes, rotation=0, fontsize=7)
        ax.set_zlim(0, zmax)
        # 0..frame までのデータを表示
        f = max(1, frame + 1)
        if st.get("wireframe"):
            surf = ax.plot_wireframe(
                X[:f, :], Y[:f, :], Z[:f, :],
                rstride=st.get("rstride", 1), cstride=st.get("cstride", 1),
                linewidth=st.get("linewidth", 0.6), color=st.get("edgecolor", "#666666"),
            )
        else:
            surf = ax.plot_surface(
                X[:f, :], Y[:f, :], Z[:f, :],
                cmap=st.get("cmap", "viridis"), edgecolor=st.get("edgecolor", "none"),
                antialiased=bool(st.get("antialiased", True)), alpha=st.get("alpha", 1.0),
            )
        if rotate:
            ax.view_init(elev=base_elev, azim=(base_azim + frame * 2) % 360)
        else:
            ax.view_init(elev=base_elev, azim=base_azim)
        return [surf]

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=1000 // max(1, fps), blit=False)

    # ライター選択
    saved_path = outfile
    try:
        if ext == ".mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
                ani.save(outfile, writer=writer, dpi=int(st.get("dpi", 120)))
            except Exception:
                # ffmpegが無い場合はgifにフォールバック
                saved_path = root + ".gif"
                writer = animation.PillowWriter(fps=fps)
                ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
        elif ext in (".gif", ".webm"):
            # webmはサポートが限定的なのでgifに寄せる
            if ext == ".webm":
                saved_path = root + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
        else:
            # 未指定ならgif
            saved_path = root + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
    finally:
        plt.close(fig)

    return saved_path


# ---- 追加の3D可視化（最小実装） ----
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_leadtime_surface_png(
    metrics: Dict[str, Any],
    outfile: str,
    title: str = "Lead Time Decomposition (3D)",
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """平均作業/待機/意思決定遅延を簡易3Dサーフェスで可視化（時間方向はダミー）。"""
    _ensure_dir(outfile)

    loss_time = ((metrics or {}).get("loss") or {}).get("time") or {}
    avg_work = float(loss_time.get("primary", {}).get("avg_work", 0.0))
    avg_wait = float(loss_time.get("primary", {}).get("avg_wait", 0.0))
    avg_dec = float(loss_time.get("primary", {}).get("avg_decision", 0.0))

    # 時間方向にダミーの軸（0..9）を置き、Y軸を要素（Wait/Work/Decision）とする
    times = np.arange(10)
    comps = ["WAIT", "WORK", "DECISION"]
    vals = np.array([avg_wait, avg_work, avg_dec], dtype=float)
    X = np.tile(times.reshape(-1, 1), (1, len(comps)))
    Y = np.tile(np.arange(len(comps)).reshape(1, -1), (len(times), 1))
    Z = np.tile(vals.reshape(1, -1), (len(times), 1))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)
    _apply_common(ax, fig, st)
    if st.get("wireframe"):
        surf = ax.plot_wireframe(
            X, Y, Z,
            rstride=st.get("rstride", 1), cstride=st.get("cstride", 1),
            linewidth=st.get("linewidth", 0.6), color=st.get("edgecolor", "#666666"),
        )
    else:
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=st.get("cmap", "plasma"), edgecolor=st.get("edgecolor", "none"),
            antialiased=bool(st.get("antialiased", True)), alpha=st.get("alpha", 1.0),
        )
    ax.set_title(title)
    ax.set_xlabel('Time (dummy)')
    ax.set_ylabel('Component')
    ax.set_zlabel('Avg Days')
    ax.set_yticks(np.arange(len(comps)))
    ax.set_yticklabels(comps, fontsize=8)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else 25, azim=azim if azim is not None else 45)
    if st.get("colorbar", True) and not st.get("wireframe"):
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    fig.savefig(outfile, dpi=int(st.get("dpi", 120)))
    plt.close(fig)
    return outfile


def render_leadtime_surface_video(
    metrics: Dict[str, Any],
    outfile: str,
    fps: int = 10,
    rotate: bool = True,
    frame_step: int = 1,
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """上記の簡易サーフェスを回転アニメーションとして保存。"""
    _ensure_dir(outfile)
    loss_time = ((metrics or {}).get("loss") or {}).get("time") or {}
    avg_work = float(loss_time.get("primary", {}).get("avg_work", 0.0))
    avg_wait = float(loss_time.get("primary", {}).get("avg_wait", 0.0))
    avg_dec = float(loss_time.get("primary", {}).get("avg_decision", 0.0))

    times = np.arange(10)
    comps = ["WAIT", "WORK", "DECISION"]
    vals = np.array([avg_wait, avg_work, avg_dec], dtype=float)
    X = np.tile(times.reshape(-1, 1), (1, len(comps)))
    Y = np.tile(np.arange(len(comps)).reshape(1, -1), (len(times), 1))
    Z = np.tile(vals.reshape(1, -1), (len(times), 1))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)

    base_elev = 25 if elev is None else float(elev)
    base_azim = 0 if azim is None else float(azim)

    def init():
        ax.clear()
        _apply_common(ax, fig, st)
        ax.set_xlabel('Time (dummy)')
        ax.set_ylabel('Component')
        ax.set_zlabel('Avg Days')
        ax.set_yticks(np.arange(len(comps)))
        ax.set_yticklabels(comps, fontsize=8)
        ax.view_init(elev=base_elev, azim=base_azim)
        return []

    def update(frame):
        ax.clear()
        if st.get("wireframe"):
            surf = ax.plot_wireframe(
                X, Y, Z,
                rstride=st.get("rstride", 1), cstride=st.get("cstride", 1),
                linewidth=st.get("linewidth", 0.6), color=st.get("edgecolor", "#666666"),
            )
        else:
            surf = ax.plot_surface(
                X, Y, Z,
                cmap=st.get("cmap", "plasma"), edgecolor=st.get("edgecolor", "none"),
                antialiased=bool(st.get("antialiased", True)), alpha=st.get("alpha", 1.0),
            )
        ax.set_xlabel('Time (dummy)')
        ax.set_ylabel('Component')
        ax.set_zlabel('Avg Days')
        ax.set_yticks(np.arange(len(comps)))
        ax.set_yticklabels(comps, fontsize=8)
        if rotate:
            ax.view_init(elev=base_elev, azim=(base_azim + frame * 2) % 360)
        else:
            ax.view_init(elev=base_elev, azim=base_azim)
        return [surf]

    # 静的データなのでフレーム数は一定。frame_stepで粗くする
    total_frames = max(30, 120 // max(1, int(frame_step or 1)))
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=1000 // max(1, fps), blit=False)

    root, ext = os.path.splitext(outfile)
    ext = ext.lower()
    saved_path = outfile
    try:
        if ext == ".mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
                ani.save(outfile, writer=writer, dpi=int(st.get("dpi", 120)))
            except Exception:
                saved_path = root + ".gif"
                writer = animation.PillowWriter(fps=fps)
                ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
        else:
            saved_path = root + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
    finally:
        plt.close(fig)
    return saved_path


def save_tech_diffusion_scatter_png(
    tech_history: List[Dict[str, Any]],
    outfile: str,
    title: str = "Tech Maturity Diffusion (3D)",
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """技術成熟(=y)×不確実性(=z)×時間(=x)の3D散布。"""
    _ensure_dir(outfile)
    if not tech_history:
        # 最低限の空画像
        fig = plt.figure(figsize=(6, 4))
        fig.suptitle("No tech history")
        fig.savefig(outfile, dpi=120)
        plt.close(fig)
        return outfile

    times = [float(h.get('time', 0.0)) for h in tech_history]
    names = list((tech_history[0] or {}).get('tech_items', {}).keys()) if tech_history else []

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)
    _apply_common(ax, fig, st)
    palette = st.get("palette")
    color_cycle = None
    if isinstance(palette, (list, tuple)) and len(palette) > 0:
        color_cycle = list(palette)
    for i, name in enumerate(names):
        ys = [float(h['tech_items'][name]['maturity']) for h in tech_history]
        zs = [float(h['tech_items'][name]['uncertainty']) for h in tech_history]
        xs = times
        color = None
        if color_cycle:
            color = color_cycle[i % len(color_cycle)]
        ax.plot(xs, ys, zs, label=name, color=color)
    ax.set_xlabel('Time')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Uncertainty')
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else 25, azim=azim if azim is not None else 45)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    fig.savefig(outfile, dpi=int(st.get("dpi", 120)))
    plt.close(fig)
    return outfile


def render_tech_diffusion_video(
    tech_history: List[Dict[str, Any]],
    outfile: str,
    fps: int = 10,
    rotate: bool = True,
    frame_step: int = 1,
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    _ensure_dir(outfile)
    times = [float(h.get('time', 0.0)) for h in (tech_history or [])]
    names = list((tech_history[0] or {}).get('tech_items', {}).keys()) if tech_history else []

    # frame_stepで時刻を間引く
    try:
        step = max(1, int(frame_step or 1))
    except Exception:
        step = 1
    if step > 1 and times:
        idx = np.arange(0, len(times), step, dtype=int)
        if len(idx) == 0 or (idx[-1] != len(times) - 1):
            idx = np.append(idx, len(times) - 1)
        tech_history = [tech_history[i] for i in idx]
        times = [times[i] for i in idx]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)
    
    base_elev = 25 if elev is None else float(elev)
    base_azim = 0 if azim is None else float(azim)

    def init():
        ax.clear()
        _apply_common(ax, fig, st)
        ax.set_xlabel('Time')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Uncertainty')
        ax.view_init(elev=base_elev, azim=base_azim)
        return []

    def update(frame):
        ax.clear()
        _apply_common(ax, fig, st)
        palette = st.get("palette")
        color_cycle = None
        if isinstance(palette, (list, tuple)) and len(palette) > 0:
            color_cycle = list(palette)
        for i, name in enumerate(names):
            ys = [float(h['tech_items'][name]['maturity']) for h in tech_history[:frame+1]]
            zs = [float(h['tech_items'][name]['uncertainty']) for h in tech_history[:frame+1]]
            xs = [float(h.get('time', 0.0)) for h in tech_history[:frame+1]]
            color = None
            if color_cycle:
                color = color_cycle[i % len(color_cycle)]
            ax.plot(xs, ys, zs, label=name, color=color)
        if rotate:
            ax.view_init(elev=base_elev, azim=(base_azim + frame * 2) % 360)
        else:
            ax.view_init(elev=base_elev, azim=base_azim)
        ax.set_xlabel('Time')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Uncertainty')
        return []

    frames = max(1, len(times))
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000 // max(1, fps), blit=False)

    root, ext = os.path.splitext(outfile)
    ext = ext.lower()
    saved_path = outfile
    try:
        if ext == ".mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
                ani.save(outfile, writer=writer, dpi=int(st.get("dpi", 120)))
            except Exception:
                saved_path = root + ".gif"
                writer = animation.PillowWriter(fps=fps)
                ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
        else:
            saved_path = root + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
    finally:
        plt.close(fig)
    return saved_path


def save_network3d_png(
    wip_history: List[Dict[str, Any]] | None,
    outfile: str,
    title: str = "Department Network (3D)",
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    """最小ネットワーク図（ノードのみを円配置、エッジは簡易連結）。"""
    _ensure_dir(outfile)
    # ノード集合をwip_historyから抽出
    nodes = []
    if wip_history:
        node_ids = set()
        for r in wip_history:
            node_ids.update((r.get('node_wip') or {}).keys())
        nodes = sorted(node_ids)
    # なければダミー
    if not nodes:
        nodes = ["DeptA", "DeptB", "DeptC"]

    n = len(nodes)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)
    zs = np.linspace(0.2, 0.8, n)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)
    _apply_common(ax, fig, st)
    ax.scatter(xs, ys, zs, s=st.get("node_size", 80), c=st.get("node_color", "steelblue"), edgecolors=st.get("node_edgecolor"))
    text_color = st.get("text_color")
    for i, name in enumerate(nodes):
        ax.text(xs[i], ys[i], zs[i]+0.03, name, fontsize=8, color=text_color)
    # 簡易エッジ（隣接を結ぶ）
    for i in range(n):
        j = (i + 1) % n
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], color=st.get("edge_color", 'gray'), alpha=st.get("edge_alpha", 0.6), linewidth=2)
    ax.set_title(title)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else 20, azim=azim if azim is not None else 45)
    if st.get("axis_off", True):
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(outfile, dpi=int(st.get("dpi", 120)))
    plt.close(fig)
    return outfile


def render_network3d_video(
    wip_history: List[Dict[str, Any]] | None,
    outfile: str,
    fps: int = 10,
    rotate: bool = True,
    frame_step: int = 1,
    elev: float | None = None,
    azim: float | None = None,
    style: str | Dict[str, Any] | None = None,
) -> str:
    _ensure_dir(outfile)
    nodes = []
    if wip_history:
        node_ids = set()
        for r in wip_history:
            node_ids.update((r.get('node_wip') or {}).keys())
        nodes = sorted(node_ids)
    if not nodes:
        nodes = ["DeptA", "DeptB", "DeptC"]

    n = len(nodes)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)
    zs = np.linspace(0.2, 0.8, n)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    st = _resolve_style(style)

    def draw_graph():
        _apply_common(ax, fig, st)
        ax.scatter(xs, ys, zs, s=st.get("node_size", 80), c=st.get("node_color", "steelblue"), edgecolors=st.get("node_edgecolor"))
        text_color = st.get("text_color")
        for i, name in enumerate(nodes):
            ax.text(xs[i], ys[i], zs[i]+0.03, name, fontsize=8, color=text_color)
        for i in range(n):
            j = (i + 1) % n
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], color=st.get("edge_color", 'gray'), alpha=st.get("edge_alpha", 0.6), linewidth=2)

    base_elev = 20 if elev is None else float(elev)
    base_azim = 0 if azim is None else float(azim)

    def init():
        ax.clear()
        draw_graph()
        if not rotate:
            ax.view_init(elev=base_elev, azim=base_azim)
        else:
            ax.view_init(elev=base_elev, azim=base_azim)
        if st.get("axis_off", True):
            ax.set_axis_off()
        return []

    def update(frame):
        ax.clear()
        draw_graph()
        if rotate:
            ax.view_init(elev=base_elev, azim=(base_azim + frame * 2) % 360)
        else:
            ax.view_init(elev=base_elev, azim=base_azim)
        ax.set_axis_off()
        return []

    total_frames = max(30, 120 // max(1, int(frame_step or 1)))
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=1000 // max(1, fps), blit=False)

    root, ext = os.path.splitext(outfile)
    ext = ext.lower()
    saved_path = outfile
    try:
        if ext == ".mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
                ani.save(outfile, writer=writer, dpi=int(st.get("dpi", 120)))
            except Exception:
                saved_path = root + ".gif"
                writer = animation.PillowWriter(fps=fps)
                ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
        else:
            saved_path = root + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(saved_path, writer=writer, dpi=int(st.get("dpi_gif", 100)))
    finally:
        plt.close(fig)
    return saved_path
