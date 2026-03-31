
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ───────── tiny DTW helper ───────────────────────────────────
def dtw_distance(a, b, window=None):
    n, m = len(a), len(b)
    inf  = float("inf")
    if window is None: window = max(n, m)
    window = max(window, abs(n - m))
    dtw = np.full((n + 1, m + 1), inf); dtw[0,0] = 0.
    for i in range(1, n + 1):
        jmin, jmax = max(1, i - window), min(m, i + window)
        for j in range(jmin, jmax + 1):
            cost = abs(a[i-1] - b[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    return dtw[n,m]

# ───────── utilities ─────────────────────────────────────────
def on_unit_sphere(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else np.array([1.,0.,0.])

def orthonormalise(v, r):
    R,_ = orthogonal_procrustes(v[:min(3, len(v))], np.eye(3))
    v, r = v @ R, r @ R
    perm = np.argsort(v.var(0))[::-1]; v, r = v[:,perm], r[:,perm]
    for j in range(3):
        if np.median(v[:,j]) < 0: v[:,j] *= -1; r[:,j] *= -1
    return v, r

def kappa_from_Rbar(R): return (R*(3-R**2))/(1-R**2+1e-9)

# ───────── main optimiser ───────────────────────────────────
def estimate_velocity_from_radial_old_dtw(
    v_r,
    subset_fraction   = 0.7,
    outer_iterations  = 60,
    lambda_reg        = 0.1,
    gamma             = 0.01,
    mean_zero_velocity= False,
    tol               = 1e-4,
    random_seed       = 0,
    visualise         = True,
    true_v            = None,
    time_axis         = None,     # ← optional time stamps for animation
    max_clusters      = 8,
    camera_numbers    = None,
    dtw_window        = None,
    support_dtw_thresh= 0.15,
    support_ratio     = 0.10,
    use_support_dtw   = True,
    grid_res          = 60,
    save_anim_path    = "projection_animation.mp4"
):
    """
    Returns
    -------
    best_v, best_r, best_mask, best_loss, loss_hist, proj_images
    (projection movie saved to `save_anim_path`)
    """
    np.random.seed(random_seed)
    T, N = v_r.shape
    if time_axis is None: time_axis = np.arange(T)
    k_keep = max(1, int(round(subset_fraction * N)))
    camera_numbers = np.arange(N) if camera_numbers is None else np.asarray(camera_numbers)

    # ---- DTW support mask (optional) ------------------------
    if use_support_dtw:
        norm = T
        support_min = int(np.ceil(support_ratio * N))
        support_cnt = np.zeros(N, int)
        for i in tqdm(range(N-1), desc="Computing DTW support", ncols=100):
            for j in range(i+1, N):
                d = dtw_distance(v_r[:,i], v_r[:,j], dtw_window) / norm
                if d <= support_dtw_thresh:
                    support_cnt[i] += 1; support_cnt[j] += 1
        support_mask_static = support_cnt >= support_min
    else:
        support_mask_static = np.ones(N, bool)

    # ---- initialisation ------------------------------------
    r = np.array([on_unit_sphere(np.random.randn(3)) for _ in range(N)])
    L = np.diag([gamma]*3) + lambda_reg*np.eye(3)
    mask = np.ones(N,bool)
    best_loss = np.inf
    best_v = best_r = best_mask = None
    loss_hist = []

    pbar = tqdm(range(outer_iterations), desc="Optimising", ncols=100)
    for it in pbar:
        idx = np.where(mask)[0]

        # LS velocity estimate
        v = np.zeros((T,3))
        for t in range(T):
            A, b = r[idx], v_r[t, idx]
            AtA = A.T@A + L
            v[t] = np.linalg.solve(AtA, A.T@b) if np.linalg.det(AtA)>1e-9 \
                    else np.linalg.lstsq(AtA, A.T@b, rcond=None)[0]
        if mean_zero_velocity: v -= v.mean(0, keepdims=True)

        # camera directions
        for i in idx: r[i] = on_unit_sphere(np.linalg.lstsq(v, v_r[:,i], rcond=None)[0])
        v, r = orthonormalise(v, r)

        # DTW residuals
        pred = v @ r.T
        dist = np.array([dtw_distance(v_r[:,i], pred[:,i], dtw_window) for i in range(N)])

        eligible = np.where(support_mask_static)[0]
        if eligible.size == 0: eligible = np.arange(N)
        chosen = eligible[np.argsort(dist[eligible])[:k_keep]]
        mask = np.zeros(N,bool); mask[chosen] = True
        loss = dist[mask].mean()
        loss_hist.append(loss)
        pbar.set_postfix({"realtime loss": f"{loss:.4f}"})
        if it and abs(loss_hist[-2]-loss) < tol: break
        if loss < best_loss:
            best_loss, best_v, best_r, best_mask = loss, v.copy(), r.copy(), mask.copy()

    # ---- vMF clustering ------------------------------------
    kept_ids  = np.where(best_mask)[0]
    kept_dirs = best_r[kept_ids]
    kept_AP   = camera_numbers[kept_ids] // (64)
    if kept_dirs.shape[0] >= 3:
        best_k, best_sil, lab_best = 1, -1, np.zeros(kept_dirs.shape[0])
        for k in range(2, min(max_clusters, kept_dirs.shape[0])+1):
            labels_k = KMeans(k, random_state=0).fit_predict(kept_dirs)
            sil = silhouette_score(kept_dirs, labels_k)
            if sil > best_sil:
                best_k, best_sil, lab_best = k, sil, labels_k
        labels = lab_best
    else:
        labels = np.zeros(kept_dirs.shape[0], int)
        best_k = 1

    cluster_stats=[]
    for c in range(best_k):
        idxs=np.where(labels==c)[0]
        mu=on_unit_sphere(kept_dirs[idxs].mean(0))
        κ = kappa_from_Rbar(np.linalg.norm(kept_dirs[idxs].mean(0)))
        ap,cnt = Counter(kept_AP[idxs]).most_common(1)[0]
        perc = 100*cnt/idxs.size
        cluster_stats.append((c,mu,κ,ap,perc,idxs))
        # print(f"Cluster {c}: μ={mu}, κ≈{κ:.2f}, AP{ap} ({perc:.1f}% of cluster)")

    # ---- per-AP global stats --------------------------------
    ap_ids=camera_numbers//(64)
    ap_stats=[]
    for ap in np.unique(ap_ids):
        mu=on_unit_sphere(best_r[ap_ids==ap].mean(0))
        κ=kappa_from_Rbar(np.linalg.norm(best_r[ap_ids==ap].mean(0)))
        ap_stats.append((ap,mu,κ))
        # print(f"AP{ap}: overall μ={mu}, κ≈{κ:.2f}")

    # ---- equirectangular grid projections -------------------
    lat = np.linspace(0, np.pi, grid_res, endpoint=False) + np.pi/(2*grid_res)
    lon = np.linspace(0, 2*np.pi, 2*grid_res, endpoint=False) + np.pi/(2*grid_res)
    theta, phi = np.meshgrid(lat, lon, indexing='ij')
    dirs = np.column_stack([np.sin(theta.ravel()) * np.cos(phi.ravel()),
                            np.sin(theta.ravel()) * np.sin(phi.ravel()),
                            np.cos(theta.ravel())])       # (M,3), M = grid_res*2*grid_res
    proj_images = np.zeros((T, grid_res, 2*grid_res))
    for t in range(T):
        proj_images[t] = (best_v[t] @ dirs.T).reshape(grid_res, 2*grid_res)
    clusters_sig = []
    for (cid,_,_,ap,perc,idxs) in cluster_stats:
        obs  = v_r[:, kept_ids[idxs]].mean(1); pred=(best_v@best_r[kept_ids[idxs]].T).mean(1)
        clusters_sig.append(obs)
    # ---- visualisation --------------------------------------
    if visualise:
        plt.figure(figsize=(6,3))
        plt.plot(loss_hist,'-o'); plt.grid(); plt.title("DTW loss"); plt.show()

        plt.figure(figsize=(8,3))
        [plt.plot(best_v[:,d],label=f"v_{l}") for d,l in enumerate("xyz")]
        plt.legend(); plt.grid(); plt.title("Velocity components"); plt.show()

        plt.figure(figsize=(8,3))
        plt.plot((best_v**2).sum(1),label="‖v_est‖²")
        if true_v is not None and true_v.shape == best_v.shape:
            plt.plot((true_v**2).sum(1),'--',label="‖v_true‖²")
        plt.legend(); plt.grid(); plt.title("Energy envelope"); plt.show()

        fig, axs = plt.subplots(len(cluster_stats),1,figsize=(9,3*len(cluster_stats)),sharex=True)
        if len(cluster_stats)==1: axs=[axs]
        for ax,(cid,_,_,ap,perc,idxs) in zip(axs,cluster_stats):
            obs  = v_r[:, kept_ids[idxs]].mean(1); pred=(best_v@best_r[kept_ids[idxs]].T).mean(1)
            ax.plot(obs,label="obs"); ax.plot(pred,label="pred")
            ax.set_title(f"Cluster {cid} (dom Ant{ap}, {perc:.0f}%)"); ax.grid(); ax.legend()
        plt.xlabel("time"); plt.show()

        # cluster sphere
        u,vang=np.mgrid[0:2*np.pi:60j,0:np.pi:30j]
        fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(np.cos(u)*np.sin(vang),np.sin(u)*np.sin(vang),np.cos(vang),
                        alpha=.1,color='gray',linewidth=0)
        ax.scatter(kept_dirs[:,0],kept_dirs[:,1],kept_dirs[:,2],
                    c=cm.tab10(labels%10),s=30)
        κmax=max(s[2] for s in cluster_stats)+1e-9
        for cid,mu,κ,ap,perc,_ in cluster_stats:
            ax.quiver(0,0,0,*mu,length=1,color='k',linewidth=2+4*κ/κmax)
            ax.text(*(1.08*mu),f"κ={κ:.1f}\nAnt{ap} {perc:.0f}%",ha='center')
        ax.set_title("vMF clusters"); ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
        plt.tight_layout(); plt.show()

        # per-AP coloured sphere
        fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(np.cos(u)*np.sin(vang),np.sin(u)*np.sin(vang),np.cos(vang),
                        alpha=.1,color='gray',linewidth=0)
        cmap=plt.get_cmap('tab20'); κmax_ap=max(s[2] for s in ap_stats)+1e-9; legend=[]
        for i,(ap,mu,κ) in enumerate(ap_stats):
            col=cmap(i%20)
            ax.quiver(0,0,0,*mu,length=1,color=col,linewidth=2+4*κ/κmax_ap)
            ax.text(*(1.08*mu),f"κ={κ:.1f}",color=col,ha='center')
            legend.append(Line2D([0],[0],color=col,lw=3,label=f"Ant{ap}"))
        ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.05,1))
        ax.set_title("Per-Ant mean directions"); ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
        plt.tight_layout(); plt.show()

        # # animated projection map
        # fig_anim, ax_anim = plt.subplots(figsize=(8,4))
        # im = ax_anim.imshow(proj_images[0], cmap='seismic', origin='upper',
        #                     extent=[0,360,-90,90], vmin=proj_images.min(), vmax=proj_images.max(),
        #                     aspect='auto')
        # ax_anim.set_xlabel("longitude °"); ax_anim.set_ylabel("latitude °")
        # cbar=fig_anim.colorbar(im, ax=ax_anim, label='v·d')
        # txt = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, color='white',
        #                     bbox=dict(facecolor='black', alpha=0.5, pad=3))

        # def update(frame):
        #     im.set_data(proj_images[frame])
        #     txt.set_text(f"t = {time_axis[frame]}")
        #     return [im, txt]

        # anim = FuncAnimation(fig_anim, update, frames=T, interval=100, blit=True)
        # print(f"Saving animation to {save_anim_path} …")
        # writer = FFMpegWriter(fps=10, bitrate=1800)
        # anim.save("./" + save_anim_path, writer= 'imagemagick' )
        # plt.show()

    return best_v, best_r, best_mask, best_loss, loss_hist, proj_images, np.array(clusters_sig)
