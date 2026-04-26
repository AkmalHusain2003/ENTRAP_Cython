# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt

cnp.import_array()


cdef class EntropyProgress:
    cdef public int cluster_id
    cdef public cnp.ndarray candidate_indices
    cdef public cnp.ndarray mahalanobis_distances
    cdef public cnp.ndarray entropy_values
    cdef public int knee_index
    cdef public int n_accepted
    cdef public int cluster_size_initial
    cdef public int cluster_size_final
    cdef public object accepted_indices
    cdef public object rejected_indices

    def __init__(self,
                 int cluster_id,
                 cnp.ndarray candidate_indices,
                 cnp.ndarray mahalanobis_distances,
                 cnp.ndarray entropy_values,
                 int knee_index,
                 int n_accepted,
                 int cluster_size_initial,
                 int cluster_size_final,
                 accepted_indices=None,
                 rejected_indices=None):
        self.cluster_id = cluster_id
        self.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        self.mahalanobis_distances = np.asarray(mahalanobis_distances, dtype=np.float64)
        self.entropy_values = np.asarray(entropy_values, dtype=np.float64)
        self.knee_index = knee_index
        self.n_accepted = n_accepted
        self.cluster_size_initial = cluster_size_initial
        self.cluster_size_final = cluster_size_final
        self.accepted_indices = accepted_indices
        self.rejected_indices = rejected_indices


cdef class EntropyProgressTracker:
    cdef dict progress_data
    cdef object _lock

    def __init__(self):
        self.progress_data = {}
        self._lock = None

    cdef _ensure_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()

    def record_cluster_progress(self,
                                 int cluster_id,
                                 cnp.ndarray candidate_indices,
                                 cnp.ndarray mahalanobis_distances,
                                 cnp.ndarray entropy_values,
                                 int knee_index,
                                 int cluster_size_initial,
                                 int cluster_size_final):
        cdef int n_accepted = min(knee_index, len(candidate_indices))
        accepted_indices = candidate_indices[:n_accepted].copy()
        rejected_indices = candidate_indices[n_accepted:].copy()

        progress = EntropyProgress(
            cluster_id=cluster_id,
            candidate_indices=candidate_indices,
            mahalanobis_distances=mahalanobis_distances,
            entropy_values=entropy_values,
            knee_index=knee_index,
            n_accepted=n_accepted,
            cluster_size_initial=cluster_size_initial,
            cluster_size_final=cluster_size_final,
            accepted_indices=accepted_indices,
            rejected_indices=rejected_indices,
        )
        self._ensure_lock()
        with self._lock:
            self.progress_data[cluster_id] = progress

    def get_cluster_progress(self, int cluster_id):
        return self.progress_data.get(cluster_id, None)

    def list_clusters(self):
        return sorted(self.progress_data.keys())

    def get_accepted_indices(self, int cluster_id):
        p = self.get_cluster_progress(cluster_id)
        return None if p is None else p.accepted_indices

    def get_rejected_indices(self, int cluster_id):
        p = self.get_cluster_progress(cluster_id)
        return None if p is None else p.rejected_indices

    def get_rejected_details(self, int cluster_id):
        cdef int n_accepted, n_total
        p = self.get_cluster_progress(cluster_id)
        if p is None or p.rejected_indices is None:
            return None
        n_accepted = p.n_accepted
        n_total = len(p.candidate_indices)
        return {
            'indices': p.rejected_indices,
            'mahalanobis_distances': p.mahalanobis_distances[n_accepted:],
            'entropy_values': p.entropy_values[n_accepted:],
            'count': len(p.rejected_indices),
            'percentage': 100.0 * len(p.rejected_indices) / max(n_total, 1),
        }

    def get_accepted_details(self, int cluster_id):
        cdef int n_accepted, n_total
        p = self.get_cluster_progress(cluster_id)
        if p is None or p.accepted_indices is None:
            return None
        n_accepted = p.n_accepted
        n_total = len(p.candidate_indices)
        return {
            'indices': p.accepted_indices,
            'mahalanobis_distances': p.mahalanobis_distances[:n_accepted],
            'entropy_values': p.entropy_values[:n_accepted],
            'count': len(p.accepted_indices),
            'percentage': 100.0 * len(p.accepted_indices) / max(n_total, 1),
        }

    def plot_rejected_analysis(self, int cluster_id, figsize=(16, 6), save_path=None):
        cdef int n_accepted, n_total, n_rejected
        p = self.get_cluster_progress(cluster_id)
        if p is None:
            raise ValueError(
                f"Cluster {cluster_id} tidak ditemukan. Available: {self.list_clusters()}"
            )
        if p.rejected_indices is None or len(p.rejected_indices) == 0:
            print(f"Cluster {cluster_id}: Tidak ada kandidat yang ditolak")
            return

        n_accepted = p.n_accepted
        n_total = len(p.candidate_indices)
        n_rejected = len(p.rejected_indices)

        accepted_mahal = p.mahalanobis_distances[:n_accepted]
        rejected_mahal = p.mahalanobis_distances[n_accepted:]
        accepted_entropy = p.entropy_values[:n_accepted]
        rejected_entropy = p.entropy_values[n_accepted:]

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        bins = np.linspace(min(p.mahalanobis_distances.min(), 0),
                           p.mahalanobis_distances.max(), 30)

        ax1 = axes[0]
        ax1.hist(accepted_mahal, bins=bins, alpha=0.6, color='green',
                 label=f'Accepted (n={n_accepted})', edgecolor='black')
        ax1.hist(rejected_mahal, bins=bins, alpha=0.6, color='red',
                 label=f'Rejected (n={n_rejected})', edgecolor='black')
        if len(accepted_mahal) > 0:
            ax1.axvline(accepted_mahal.max(), color='darkgreen', linestyle='--',
                        linewidth=2, label=f'Max Accepted: {accepted_mahal.max():.3f}')
        if len(rejected_mahal) > 0:
            ax1.axvline(rejected_mahal.min(), color='darkred', linestyle='--',
                        linewidth=2, label=f'Min Rejected: {rejected_mahal.min():.3f}')
        ax1.set_xlabel('Mahalanobis Distance', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'Cluster {cluster_id}: Mahalanobis Distance\nAccepted vs Rejected',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')

        bins_e = np.linspace(min(p.entropy_values.min(), 0),
                             p.entropy_values.max(), 30)
        ax2 = axes[1]
        ax2.hist(accepted_entropy, bins=bins_e, alpha=0.6, color='green',
                 label=f'Accepted (n={n_accepted})', edgecolor='black')
        ax2.hist(rejected_entropy, bins=bins_e, alpha=0.6, color='red',
                 label=f'Rejected (n={n_rejected})', edgecolor='black')
        if len(accepted_entropy) > 0:
            ax2.axvline(accepted_entropy.max(), color='darkgreen', linestyle='--',
                        linewidth=2, label=f'Max Accepted: {accepted_entropy.max():.3f}')
        if len(rejected_entropy) > 0:
            ax2.axvline(rejected_entropy.min(), color='darkred', linestyle='--',
                        linewidth=2, label=f'Min Rejected: {rejected_entropy.min():.3f}')
        ax2.set_xlabel('Persistence Entropy', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'Cluster {cluster_id}: Persistence Entropy\nAccepted vs Rejected',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')

        ax3 = axes[2]
        ax3.scatter(accepted_mahal, accepted_entropy, c='green', s=50, alpha=0.6,
                    label=f'Accepted (n={n_accepted})', edgecolors='black', linewidths=0.5)
        ax3.scatter(rejected_mahal, rejected_entropy, c='red', s=50, alpha=0.6,
                    label=f'Rejected (n={n_rejected})', edgecolors='black', linewidths=0.5)
        if n_accepted > 0 and n_rejected > 0:
            ax3.scatter([accepted_mahal.max()], [accepted_entropy[-1]],
                        c='darkred', s=300, marker='*', edgecolors='black',
                        linewidths=2, zorder=10, label='Knee Point')
        ax3.set_xlabel('Mahalanobis Distance', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Persistence Entropy', fontsize=11, fontweight='bold')
        ax3.set_title(f'Cluster {cluster_id}: Mahalanobis vs Entropy\nDecision Boundary',
                      fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} - REJECTED CANDIDATES ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Candidates      : {n_total}")
        print(f"Accepted              : {n_accepted} ({100*n_accepted/n_total:.1f}%)")
        print(f"Rejected              : {n_rejected} ({100*n_rejected/n_total:.1f}%)")
        if len(accepted_mahal) > 0:
            print("\nACCEPTED STATS:")
            print(f"  Mahalanobis Range   : [{accepted_mahal.min():.3f}, {accepted_mahal.max():.3f}]")
            print(f"  Mahalanobis Mean    : {accepted_mahal.mean():.3f} \u00b1 {accepted_mahal.std():.3f}")
            print(f"  Entropy Range       : [{accepted_entropy.min():.3f}, {accepted_entropy.max():.3f}]")
            print(f"  Entropy Mean        : {accepted_entropy.mean():.3f} \u00b1 {accepted_entropy.std():.3f}")
        if len(rejected_mahal) > 0:
            print("\nREJECTED STATS:")
            print(f"  Mahalanobis Range   : [{rejected_mahal.min():.3f}, {rejected_mahal.max():.3f}]")
            print(f"  Mahalanobis Mean    : {rejected_mahal.mean():.3f} \u00b1 {rejected_mahal.std():.3f}")
            print(f"  Entropy Range       : [{rejected_entropy.min():.3f}, {rejected_entropy.max():.3f}]")
            print(f"  Entropy Mean        : {rejected_entropy.mean():.3f} \u00b1 {rejected_entropy.std():.3f}")
        print(f"{'='*60}\n")

    def plot_entropy_curve(self, int cluster_id, figsize=(10, 6), save_path=None):
        cdef int n_candidates, knee_index
        p = self.get_cluster_progress(cluster_id)
        if p is None:
            raise ValueError(
                f"Cluster {cluster_id} tidak ditemukan. Available: {self.list_clusters()}"
            )
        n_candidates = len(p.entropy_values)
        if n_candidates == 0:
            print(f"Cluster {cluster_id}: Tidak ada kandidat untuk divisualisasikan")
            return

        knee_index = p.knee_index
        x = np.arange(1, n_candidates + 1)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, p.entropy_values, 'b-', linewidth=2, label='Persistence Entropy', alpha=0.7)
        ax.scatter(x, p.entropy_values, c='blue', s=30, alpha=0.5, zorder=3)
        if knee_index > 0:
            ax.axvspan(0, knee_index, alpha=0.15, color='green', label='Accepted Region')
        if knee_index < n_candidates:
            ax.axvspan(knee_index, n_candidates, alpha=0.15, color='red', label='Rejected Region')
        if 0 < knee_index <= n_candidates:
            ax.axvline(x=knee_index, color='darkred', linestyle='--', linewidth=2, alpha=0.8,
                       label='Knee Point')
            ax.scatter([knee_index], [p.entropy_values[knee_index - 1]], c='darkred',
                       s=200, marker='*', edgecolors='black', linewidth=1.5, zorder=5)
        ax.set_xlabel('Candidate Order (Nearest \u2192 Farthest by Mahalanobis)',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Persistence Entropy (H\u2080)', fontsize=11, fontweight='bold')
        ax.set_title(f'Cluster {cluster_id}: Sequential Persistence Entropy',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_comparison(self, list cluster_ids, figsize=(16, 10), save_path=None):
        cdef int n_clusters, n_cols, n_rows, idx, n_candidates, knee_index
        available = [cid for cid in cluster_ids if cid in self.progress_data]
        if not available:
            print(f"Tidak ada cluster valid. Available: {self.list_clusters()}")
            return
        n_clusters = len(available)
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1) if n_clusters > 1 else [axes]
        for idx in range(n_clusters):
            cluster_id = available[idx]
            ax = axes[idx]
            p = self.progress_data[cluster_id]
            n_candidates = len(p.entropy_values)
            knee_index = p.knee_index
            x = np.arange(1, n_candidates + 1)
            ax.plot(x, p.entropy_values, 'b-', linewidth=1.5, alpha=0.7)
            ax.scatter(x, p.entropy_values, c='blue', s=20, alpha=0.4)
            if knee_index > 0:
                ax.axvspan(0, knee_index, alpha=0.15, color='green')
            if 0 < knee_index <= n_candidates:
                ax.axvline(x=knee_index, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.scatter([knee_index], [p.entropy_values[knee_index - 1]],
                           c='red', s=100, marker='*', edgecolors='darkred', zorder=5)
            ax.set_xlabel('Candidate Order', fontsize=9)
            ax.set_ylabel('Persistence Entropy', fontsize=9)
            ax.set_title(f'Cluster {cluster_id}\nAccepted: {p.n_accepted}/{n_candidates}',
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle=':')
        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')
        plt.suptitle('Entropy Curves Comparison Across Clusters',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def export_summary(self):
        cdef int cluster_id
        summary = {}
        for cluster_id, p in self.progress_data.items():
            n_cands = len(p.candidate_indices)
            summary[cluster_id] = {
                'n_candidates': n_cands,
                'n_accepted': p.n_accepted,
                'acceptance_rate': p.n_accepted / max(n_cands, 1),
                'knee_index': p.knee_index,
                'cluster_size_initial': p.cluster_size_initial,
                'cluster_size_final': p.cluster_size_final,
                'size_change': p.cluster_size_final - p.cluster_size_initial,
                'mean_entropy': float(np.mean(p.entropy_values)),
                'std_entropy': float(np.std(p.entropy_values)),
                'entropy_range': (float(np.min(p.entropy_values)),
                                  float(np.max(p.entropy_values))),
            }
        return summary
