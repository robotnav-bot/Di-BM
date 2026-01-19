import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import subprocess

plt.switch_backend('Agg')

BASE_PATH = "../../eval_result/lift_pot/Di-BM/demo_clean/demo_clean/2026-01-19 19:23:57"

COLORS = [
    "#e41a1c",  
    "#377eb8",  
    "#4daf4a",  
    "#984ea3",  
    "#ff7f00",  
    "#ffff33",  
    "#a65628",  
    "#f781bf"   
]

def get_natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def process_episode(episode_name):
    video_path = os.path.join(BASE_PATH, f"{episode_name}.mp4")
    pkl_dir = os.path.join(BASE_PATH, episode_name)
    output_path = os.path.join(BASE_PATH, f"{episode_name}_vis.mp4")

    if not os.path.exists(video_path) or not os.path.exists(pkl_dir):
        print(f"[Skip] Missing files for {episode_name}")
        return

    print(f"Processing: {episode_name} ...")

    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    pkl_files.sort(key=get_natural_key)

    data_list = []
    for pkl_f in pkl_files:
        try:
            with open(pkl_f, 'rb') as f:
                d = pickle.load(f)
                if hasattr(d, 'detach'): d = d.detach()
                if hasattr(d, 'cpu'): d = d.cpu()
                if hasattr(d, 'numpy'): d = d.numpy()
                data_list.append(np.array(d).flatten())
        except Exception:
            pass

    if not data_list:
        print(f"[Skip] No valid data in {pkl_dir}")
        return

    data_matrix = np.stack(data_list)
    num_data_steps, num_experts = data_matrix.shape

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: total_frames = 1
    if fps is None or np.isnan(fps) or fps == 0: fps = 10

    plot_w = vid_w
    plot_h = vid_h
    total_w = vid_w + plot_w
    total_h = vid_h
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{total_w}x{total_h}",
        "-framerate", str(fps),
        "-i", "-", 
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
        output_path
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    fig = plt.figure(figsize=(plot_w / 100, plot_h / 100), dpi=100)
    ax = fig.add_subplot(111)

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        data_idx = int((frame_idx / total_frames) * num_data_steps)
        data_idx = min(data_idx, num_data_steps - 1)

        ax.clear()
        x_axis = np.arange(num_data_steps)

        for i in range(num_experts):
            color = COLORS[i % len(COLORS)]
            ax.plot(x_axis, data_matrix[:, i], label=f'Expert {i}', color=color, linewidth=1.5)

        ax.axvline(x=data_idx, color='gray', linestyle='--', linewidth=1, alpha=0.8)

        ax.set_title(r'$\pi(e|o)$', fontsize=12, pad=8)

        ax.set_ylim(-0.05, 1.)
        ax.set_xlim(0, num_data_steps)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.6)
        
        ax.grid(True, linestyle=':', alpha=0.4)
        
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.08)

        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            plot_img = np.asarray(buf)
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)
        except AttributeError:
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            plot_img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        if plot_img.shape[0] != plot_h or plot_img.shape[1] != plot_w:
            plot_img = cv2.resize(plot_img, (plot_w, plot_h))

        combined_img = np.hstack((frame_rgb, plot_img))
        
        process.stdin.write(combined_img.tobytes())

        if frame_idx % 50 == 0:
            print(f"  > Encoded frame {frame_idx}/{total_frames}")

    cap.release()
    plt.close(fig)
    process.stdin.close()
    process.wait()
    print(f"Saved: {output_path}\n")

if __name__ == "__main__":
    all_items = glob.glob(os.path.join(BASE_PATH, "episode*"))
    episode_dirs = [d for d in all_items if os.path.isdir(d)]
    episode_names = [os.path.basename(d) for d in episode_dirs]
    episode_names.sort(key=get_natural_key)

    for ep_name in episode_names:
        process_episode(ep_name)