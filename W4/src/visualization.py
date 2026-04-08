import os
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvironmentManager

GRID_ROWS = 4
GRID_COLS = 12
ACTION_ARROWS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # up, right, down, left


class Visualizer:
    def __init__(self, env_manager: EnvironmentManager):
        self.env = env_manager

    def plot_learning_curves(self, results: list, output_dir: str = None, ylim: tuple = None, smooth_window: int = None, filename: str = 'learning_curves', title: str = 'Learning Curves', show_learning_speed: bool = False, learning_speed_threshold: float = -75.0) -> None:
        def _smooth(arr):
            kernel = np.ones(smooth_window) / smooth_window
            return np.convolve(arr, kernel, mode='valid')

        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            matrix = result.reward_matrix
            n_seeds = matrix.shape[0]
            episodes = np.arange(matrix.shape[1])
            mean = np.mean(matrix, axis=0)
            ci = 1.96 * np.std(matrix, axis=0) / np.sqrt(n_seeds)
            if smooth_window:
                mean = _smooth(mean)
                ci   = _smooth(ci)
                episodes = episodes[:len(mean)]
            line, = ax.plot(episodes, mean, label=result.config.label)
            color = line.get_color()
            ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.2, color=color)
        if show_learning_speed and results:
            ax.axhline(y=learning_speed_threshold, color='gray', linestyle='--', linewidth=1.0, alpha=0.8, label=f'threshold ({learning_speed_threshold})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Episode Return')
        ax.set_title(title)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        if results:
            # Build annotation from parameters constant across all results
            configs = [r.config for r in results]
            alphas   = set(c.alpha for c in configs)
            scheds   = set(type(c.epsilon_schedule).__name__ for c in configs)
            seeds    = set(c.n_seeds for c in configs)
            alpha_str = f'α={next(iter(alphas))}' if len(alphas) == 1 else 'α=varied'
            sched_str = f'ε={next(iter(scheds))}' if len(scheds) == 1 else 'ε=varied'
            seeds_str = f'seeds={next(iter(seeds))}' if len(seeds) == 1 else 'seeds=varied'
            info_line1 = f'{alpha_str} | {sched_str} | {seeds_str}'
            info_line2_parts = []
            if smooth_window:
                info_line2_parts.append(f'smoothed window={smooth_window}')
            if ylim:
                ax.set_ylim(ylim)
                info_line2_parts.append(f'y clipped to {ylim}')
            info_line2_parts.append('95% CI shaded')
            info = info_line1 + '\n' + ' | '.join(info_line2_parts)
            ax.text(0.98, 0.05, info, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        elif ylim:
            ax.set_ylim(ylim)
        fig.tight_layout()
        if output_dir:
            fig.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight')

    def plot_policy_arrows(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        for state in range(GRID_ROWS * GRID_COLS):
            row, col = divmod(state, GRID_COLS)
            action = np.argmax(q_table[state])
            dx, dy = ACTION_ARROWS[action]
            ax.annotate('', xy=(col + dx * 0.3, GRID_ROWS - 1 - row + dy * 0.3),
                        xytext=(col, GRID_ROWS - 1 - row),
                        arrowprops=dict(arrowstyle='->', color='black'))
        # mark cliff
        for col in range(1, GRID_COLS - 1):
            ax.add_patch(plt.Rectangle((col - 0.5, -0.5), 1, 1, color='red', alpha=0.3))
        ax.set_xlim(-0.5, GRID_COLS - 0.5)
        ax.set_ylim(-0.5, GRID_ROWS - 0.5)
        ax.set_title(f'Policy Arrows: {label}')
        ax.set_aspect('equal')
        if output_dir:
            fig.savefig(os.path.join(output_dir, f'policy_arrows_{label}.png'))

    def plot_value_heatmap(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        values = np.max(q_table, axis=1).reshape(GRID_ROWS, GRID_COLS)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(values, cmap='RdYlGn', interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Max Q-value')
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                ax.text(col, row, f'{values[row, col]:.0f}',
                        ha='center', va='center', fontsize=6,
                        color='black')
        ax.set_title(f'Value Heatmap: {label}')
        if output_dir:
            fig.savefig(os.path.join(output_dir, f'value_heatmap_{label}.png'))

    def plot_trajectory_comparison(self, results: list, output_dir: str = None) -> None:
        colors = ['steelblue', 'darkorange', 'green', 'purple']
        fig, ax = plt.subplots(figsize=(12, 4))
        # draw grid
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                ax.add_patch(plt.Rectangle((col - 0.5, GRID_ROWS - 1 - row - 0.5), 1, 1,
                                           fill=False, edgecolor='gray'))
        # mark cliff
        for col in range(1, GRID_COLS - 1):
            ax.add_patch(plt.Rectangle((col - 0.5, -0.5), 1, 1, color='red', alpha=0.3))
        # mark start and goal
        ax.text(0, 0, 'S', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(GRID_COLS - 1, 0, 'G', ha='center', va='center', fontsize=10, fontweight='bold')
        # draw one greedy path per result
        for result, color in zip(results, colors):
            state = self.env.reset(seed=0)
            path = [state]
            for _ in range(500):
                action = int(np.argmax(result.q_table[state]))
                state, _, terminated, truncated = self.env.step(action)
                path.append(state)
                if terminated or truncated:
                    break
            cols = [s % GRID_COLS for s in path]
            rows = [GRID_ROWS - 1 - s // GRID_COLS for s in path]
            ax.plot(cols, rows, '-o', color=color, markersize=4, label=result.config.label)
        ax.set_xlim(-0.5, GRID_COLS - 0.5)
        ax.set_ylim(-0.5, GRID_ROWS - 0.5)
        ax.set_title('Greedy Trajectories: SARSA vs Q-Learning')
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), bbox_inches='tight')

    def plot_trajectory(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        # Roll out greedy policy from start state
        state = self.env.reset(seed=0)
        path = [state]
        for _ in range(200):
            action = int(np.argmax(q_table[state]))
            state, _, terminated, truncated = self.env.step(action)
            path.append(state)
            if terminated or truncated:
                break
        fig, ax = plt.subplots(figsize=(12, 4))
        # draw grid
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                ax.add_patch(plt.Rectangle((col - 0.5, GRID_ROWS - 1 - row - 0.5), 1, 1,
                                           fill=False, edgecolor='gray'))
        # mark cliff
        for col in range(1, GRID_COLS - 1):
            ax.add_patch(plt.Rectangle((col - 0.5, -0.5), 1, 1, color='red', alpha=0.3))
        # draw path
        cols = [s % GRID_COLS for s in path]
        rows = [GRID_ROWS - 1 - s // GRID_COLS for s in path]
        ax.plot(cols, rows, 'b-o', markersize=4)
        ax.set_xlim(-0.5, GRID_COLS - 0.5)
        ax.set_ylim(-0.5, GRID_ROWS - 0.5)
        ax.set_title(f'Trajectory: {label}')
        ax.set_aspect('equal')
        if output_dir:
            fig.savefig(os.path.join(output_dir, f'trajectory_{label}.png'))
