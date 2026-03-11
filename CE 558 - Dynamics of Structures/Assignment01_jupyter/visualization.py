import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class ShearBuildingVisualizer:
    """
    Provides methods for visualizing shear buildings.

    This class is not intended to be instantiated directly. Instead, it should be
    accessed through the `plot` property of an `MDF` instance that represents a
    shear building (e.g., `mdf.plot.structure()`).
    """

    def __init__(self, mdf, story_height=3.0, building_width=5.0):
        """
        Initializes the ShearBuildingVisualizer.

        Parameters
        ----------
        mdf : MDF
            The MDF object representing the shear building.
        story_height : float, optional
            The height of each story, by default 3.0.
        building_width : float, optional
            The width of the building, by default 5.0.
        """
        if not (hasattr(mdf, "masses") and hasattr(mdf, "stiffnesses")):
            raise TypeError(
                "The `plot` methods are only available for MDF objects "
                "created with `from_shear_building`."
            )
        self.mdf = mdf
        self.n_stories = len(mdf.masses)
        self.story_height = story_height
        self.building_width = building_width

    def _plot_displaced_shape(
        self, ax, displacements, title="", max_disp_override=None
    ):
        """Helper to plot the displaced shape of the building on a given axis."""
        ax.clear()
        displacements = np.insert(displacements, 0, 0)  # Add base displacement

        # Generate points for plotting smooth column curves
        num_points_curve = 20
        t_curve = np.linspace(0, 1, num_points_curve)
        # Cubic Hermite spline basis for zero-slope start/end points
        hermite_poly = -2 * t_curve**3 + 3 * t_curve**2

        # Plot columns with double curvature
        for i in range(self.n_stories):
            y1, y2 = i * self.story_height, (i + 1) * self.story_height
            x1, x2 = displacements[i], displacements[i + 1]

            y_curve = np.linspace(y1, y2, num_points_curve)
            x_curve = x1 + (x2 - x1) * hermite_poly

            # Plot left and right columns
            ax.plot(x_curve, y_curve, "b-")
            ax.plot(x_curve + self.building_width, y_curve, "b-")

        # Plot floors (as horizontal lines)
        for i in range(self.n_stories):
            y = (i + 1) * self.story_height
            x_start = displacements[i + 1]
            x_end = x_start + self.building_width
            ax.plot([x_start, x_end], [y, y], "k-", lw=2)

        # Plot ground line
        ax.axhline(0, color="k", lw=2)
        ax.plot(
            [-self.building_width, 2 * self.building_width],
            [0, 0],
            color="k",
            linestyle="--",
            lw=1,
        )

        # Formatting
        if max_disp_override is not None:
            max_disp = max_disp_override
        else:
            max_disp = np.max(np.abs(displacements))
            if max_disp == 0:  # Avoid empty plot for zero displacement
                max_disp = 1.0

        ax.set_xlim(-max_disp * 1.5 - 0.5, self.building_width + max_disp * 1.5 + 0.5)
        ax.set_ylim(-self.story_height * 0.5, (self.n_stories + 1) * self.story_height)
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        ax.set_ylabel("Story")
        ax.set_yticks(
            np.arange(self.n_stories + 1) * self.story_height,
            [f"{i}" for i in range(self.n_stories + 1)],
        )

    def structure(self):
        """Plots the undeformed structure of the shear building."""
        fig, ax = plt.subplots()
        displacements = np.zeros(self.n_stories)
        self._plot_displaced_shape(ax, displacements, title="Shear Building")
        # ax.set_xlabel("Displacement")
        plt.show()

    def mode_shape(self, mode_number=[1]):
        """
        Plots one or more specified mode shapes of the shear building.

        Parameters
        ----------
        mode_number : int or list of int, optional
            A single mode number or a list of mode numbers to plot (1-indexed).
            If not provided, defaults to plotting the first mode, i.e., `[1]`.
        """

        if isinstance(mode_number, int):
            mode_number = [mode_number]

        if not isinstance(mode_number, list) or not all(
            isinstance(n, int) for n in mode_number
        ):
            raise TypeError("mode_number must be an integer or a list of integers.")

        if self.mdf.modal.phi is None:
            self.mdf.modal.modal_analysis()

        max_mode = self.mdf.modal.phi.shape[1]
        for mn in mode_number:
            if not 1 <= mn <= max_mode:
                raise ValueError(
                    f"Invalid mode number: {mn}. Must be between 1 and {max_mode}."
                )

        num_modes_to_plot = len(mode_number)

        fig, axes = plt.subplots(
            1, num_modes_to_plot, figsize=(5 * num_modes_to_plot, 6), squeeze=False
        )
        axes = axes.flatten()

        for i, mn in enumerate(mode_number):
            omega = self.mdf.modal.omega[mn - 1]
            mode_shape_vector = self.mdf.modal.phi[:, mn - 1]
            norm_factor = np.max(np.abs(mode_shape_vector))
            normalized_shape = (
                mode_shape_vector / norm_factor
                if norm_factor != 0
                else mode_shape_vector
            )

            ax = axes[i]
            self._plot_displaced_shape(
                ax,
                normalized_shape,
                title=f"Mode {mn}",
            )
            text_str = rf"$\omega_{mn}$ = {omega:.2f} rad/s"
            ax.text(
                0.05,
                0.95,
                text_str,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
            # ax.set_xlabel("Normalized Displacement")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def animate_response(
        self,
        response_df,
        scale_factor=20,
        ground_motion=None,
        speed_up=1.0,
        repeat=True,
        save_path=None,
    ):
        """
        Animates the dynamic response of the shear building.

        Parameters
        ----------
        response_df : pandas.DataFrame
            The response DataFrame, as returned by a solver like `find_response`.
        scale_factor : int, optional
            A factor to scale the displacements for better visualization, by default 20.
        ground_motion : tuple, optional
            A tuple `(time, acceleration)` for the ground motion history. If provided,
            a plot of the ground motion will be shown below the building animation.
            By default None.
        speed_up : float, optional
            The factor by which to speed up the animation, by default 1.0.
        repeat : bool, optional
            Whether the animation should repeat when finished, by default True.
        save_path : str, optional
            The file path to save the animation (e.g., 'animation.mp4'). If provided,
            the animation is saved to file instead of being shown. By default None.
        """
        u_cols = [f"u{i + 1}" for i in range(self.n_stories)]
        displacements = response_df[u_cols].to_numpy()
        time_vector = response_df["time"].to_numpy()

        # Create subplots
        if ground_motion:
            fig, (ax_build, ax_gm) = plt.subplots(
                2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
            )
            gm_time, gm_acc = ground_motion
            ax_gm.plot(gm_time, gm_acc, "k-")
            ax_gm.set_xlabel("Time (s)")
            ax_gm.set_ylabel("Ground Acc. (g)")
            ax_gm.grid(True)
            (time_marker,) = ax_gm.plot([], [], "r-", lw=2)  # The vertical line
        else:
            fig, ax_build = plt.subplots(figsize=(8, 6))
            # ax_build.set_xlabel(
            #     "Displacement"
            # )  # Only add xlabel if no ground motion plot

        # The override should be scaled to match the scaled displacements
        max_abs_disp = np.max(np.abs(displacements)) * scale_factor

        def update(frame):
            frame_index = int(frame)
            current_time = time_vector[frame_index]

            # Get original (unscaled) roof displacement for text display
            roof_disp_unscaled = displacements[frame_index, -1]

            # Scale displacements for visualization
            current_displacements_scaled = displacements[frame_index, :] * scale_factor

            self._plot_displaced_shape(
                ax_build,
                current_displacements_scaled,
                title=f"Deformed shape (SF = {scale_factor})",
                max_disp_override=max_abs_disp,
            )

            # Add text annotation inside the plot
            text_str = (
                f"Time: {current_time:.2f} s\nRoof Disp: {roof_disp_unscaled:.4f} m"
            )
            ax_build.text(
                0.05,
                0.95,
                text_str,
                transform=ax_build.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            if ground_motion:
                # Update the time marker on the ground motion plot
                ylim = ax_gm.get_ylim()
                time_marker.set_data([current_time, current_time], [ylim[0], ylim[1]])

        # --- New robust logic for speed_up ---
        dt = time_vector[1] - time_vector[0]

        # Target 50 FPS for a smooth animation
        target_interval_ms = 20

        # Required interval to match the desired speed_up without frame skipping
        required_interval_ms = dt * 1000 / speed_up

        if required_interval_ms >= target_interval_ms:
            # For slow motion or real-time, no need to skip frames
            frame_step = 1
            interval = required_interval_ms
        else:
            # For fast motion, skip frames to maintain a smooth playback
            interval = target_interval_ms
            # Calculate how many frames to jump over
            frame_step = int(round(speed_up * interval / (dt * 1000)))
            # Ensure we are always moving forward
            frame_step = max(1, frame_step)

        frames_to_render = np.arange(0, len(time_vector), frame_step)

        anim = FuncAnimation(
            fig,
            update,
            frames=frames_to_render,
            interval=interval,
            repeat=repeat,
        )
        # Adjust layout to prevent text from being trimmed
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            print(f"Saving animation to {save_path}... This may take a moment.")
            try:
                anim.save(save_path, writer="ffmpeg", dpi=150)
                print(f"Animation successfully saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print(
                    "Please ensure FFmpeg is installed and accessible in your system's PATH."
                )
            plt.close(fig)  # Close the figure to free up memory
        else:
            plt.show()

        return anim
