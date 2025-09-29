import numpy as np
from manim import *
from manim_slides import Slide

# --- Helper Functions for the Standard Map ---

def get_standard_map_trajectory(theta0, p0, K, iterations=200):
    """Calculates a single trajectory of the standard map."""
    thetas = np.zeros(iterations)
    ps = np.zeros(iterations)
    thetas[0], ps[0] = theta0, p0
    
    for i in range(1, iterations):
        p_next = (ps[i-1] + K * np.sin(thetas[i-1]))
        t_next = (thetas[i-1] + p_next)
        
        # Bring back to the [-pi, pi] and [0, 2pi] range
        thetas[i] = t_next % (2 * PI)
        ps[i] = (p_next + PI) % (2 * PI) - PI
        
    return thetas, ps

def create_phase_space_vgroup(axes, K, samples=25, iterations=200, dot_radius=0.005):
    """Creates a VGroup of dots representing the phase space for a given K."""
    dots = VGroup()
    grid_points = np.linspace(0, 2*PI, samples)
    
    for theta0 in grid_points:
        for p0 in np.linspace(-PI, PI, samples // 2):
            thetas, ps = get_standard_map_trajectory(theta0, p0, K, iterations)
            # Add dots for this trajectory
            for t, p in zip(thetas, ps):
                dots.add(Dot(axes.c2p(t, p), radius=dot_radius, color=BLUE))
                
    return dots

# --- Manim Slides Presentation Class ---

class StandardMapTalk(Slide):
    def construct(self):
        # --- Common elements for all slides ---
        self.axes = Axes(
            x_range=[0, 2 * PI, PI / 2],
            y_range=[-PI, PI, PI / 2],
            x_length=7,
            y_length=7,
            axis_config={"color": GRAY},
        ).to_edge(RIGHT, buff=1)
        
        x_labels = {0: "0", PI: r"$\pi$", 2*PI: r"$2\pi$"}
        y_labels = {-PI: r"$-\pi$", 0: "0", PI: r"$\pi$"}
        self.axes.add_coordinates(x_labels, y_labels)
        
        self.theta_label = self.axes.get_x_axis_label(r"\theta_n", edge=DOWN, direction=DOWN)
        self.p_label = self.axes.get_y_axis_label(r"p_n", edge=LEFT, direction=LEFT)
        self.axes_group = VGroup(self.axes, self.theta_label, self.p_label)

        # Let's start the presentation
        self.slide_1_title()
        self.slide_2_model()
        self.slide_3_anatomy()
        self.slide_4_subcritical()
        self.slide_5_critical()
        self.slide_6_supercritical()
        self.slide_7_diffusion()
        self.slide_8_anomalous()
        self.slide_9_summary()
        self.slide_10_questions()

    def slide_1_title(self):
        """TITLE SLIDE"""
        title = Title("Phase Space Topology and Global Transport in the Standard Map")
        author = Text("Your Name\nYour Research Group", font_size=32).next_to(title, DOWN, buff=0.5)
        
        # Pre-render a nice phase space plot to show in the background
        k_critical_plot = create_phase_space_vgroup(self.axes, K=0.97, samples=20, iterations=150).set_opacity(0.5)

        self.play(Write(title), FadeIn(author))
        self.play(FadeIn(self.axes_group), Create(k_critical_plot))
        self.next_slide()
        self.play(FadeOut(k_critical_plot)) # Keep axes for next slide

    def slide_2_model(self):
        """THE MODEL"""
        self.play(FadeOut(self.wipe_all_but(self.axes_group))) # Keep axes
        
        title = Title("A Canonical Model for Hamiltonian Transport")
        equations = MathTex(
            r"p_{n+1} &= p_n + K \sin(\theta_n) \pmod{2\pi} \\",
            r"\theta_{n+1} &= \theta_n + p_{n+1} \pmod{2\pi}",
            font_size=48
        ).to_edge(LEFT, buff=1)
        
        k_text = Tex(r"K: Stochasticity Parameter", font_size=36).next_to(equations, DOWN, buff=1)

        self.play(Write(title))
        self.play(Write(equations))
        self.play(FadeIn(k_text, shift=UP))
        self.next_slide()
        
    def slide_3_anatomy(self):
        """PHASE SPACE ANATOMY"""
        self.play(FadeOut(self.wipe_all_but(self.axes_group)))
        
        title = Title("Phase Space Anatomy: Barriers and Conduits")
        self.play(Write(title))

        k_value = 0.8
        phase_plot = create_phase_space_vgroup(self.axes, K=k_value, dot_radius=0.006)
        k_label = MathTex(f"K = {k_value}").to_corner(UR).shift(LEFT*2)

        self.play(Create(phase_plot), Write(k_label))
        
        # Add annotations
        kam_label = Tex("KAM Tori (Barriers)").scale(0.7).next_to(self.axes.c2p(PI, 2.5), UP)
        kam_arrow = Arrow(kam_label.get_bottom(), self.axes.c2p(PI, 1.8), buff=0.1)
        
        sea_label = Tex("Stochastic Sea (Conduit)").scale(0.7).next_to(self.axes.c2p(5.5, -1), UP)
        sea_arrow = Arrow(sea_label.get_bottom(), self.axes.c2p(5.0, -0.2), buff=0.1)

        self.play(GrowArrow(kam_arrow), Write(kam_label))
        self.play(GrowArrow(sea_arrow), Write(sea_label))
        self.next_slide()
        
    def slide_4_subcritical(self):
        """SUB-CRITICAL REGIME"""
        self.play(FadeOut(self.wipe_all_but(self.axes_group)))
        title = Title("Sub-Critical ($K < K_c$): Global Confinement")
        self.play(Write(title))
        
        k_value = 0.5
        k_label = MathTex(f"K = {k_value}").to_corner(UR).shift(LEFT*2)
        phase_plot_sub = create_phase_space_vgroup(self.axes, K=k_value, samples=20, dot_radius=0.008)
        
        self.play(Create(phase_plot_sub), Write(k_label))
        self.next_slide()
        self.previous_phase_plot = phase_plot_sub # Save for transition

    def slide_5_critical(self):
        """CRITICAL POINT"""
        self.play(FadeOut(self.wipe_all_but([self.axes_group, self.previous_phase_plot])))
        title = Title("Critical Point ($K_c \\approx 0.9716$): Destruction of the Golden Torus")
        self.play(Write(title))
        
        k_value = 0.9716
        k_label = MathTex(f"K_c \\approx {k_value}").to_corner(UR).shift(LEFT*2)
        phase_plot_crit = create_phase_space_vgroup(self.axes, K=k_value, samples=20, dot_radius=0.008)

        self.play(Transform(self.previous_phase_plot, phase_plot_crit), Write(k_label))
        self.next_slide()
        self.previous_phase_plot = self.previous_phase_plot # It was transformed
        
    def slide_6_supercritical(self):
        """SUPER-CRITICAL REGIME"""
        self.play(FadeOut(self.wipe_all_but([self.axes_group, self.previous_phase_plot])))
        title = Title("Super-Critical ($K > K_c$): Global Stochasticity")
        self.play(Write(title))

        k_value = 1.2
        k_label = MathTex(f"K = {k_value}").to_corner(UR).shift(LEFT*2)
        phase_plot_super = create_phase_space_vgroup(self.axes, K=k_value, samples=20, dot_radius=0.008)

        self.play(Transform(self.previous_phase_plot, phase_plot_super), Write(k_label))
        self.next_slide()
        self.play(FadeOut(self.wipe_all_but([]))) # Clean slate for next slide

    def slide_7_diffusion(self):
        """DIFFUSION COEFFICIENT"""
        title = Title("Quantifying Transport: The Diffusion Coefficient")
        self.play(Write(title))

        diff_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 3, 1],
            x_length=10, y_length=6,
            axis_config={"include_tip": False}
        ).add_coordinates()
        
        x_ax_label = diff_axes.get_x_axis_label("K")
        y_ax_label = diff_axes.get_y_axis_label(r"D = \lim_{n\to\infty} \frac{\langle p_n^2 \rangle}{n}", direction=LEFT)
        
        # Approximate diffusion curve D vs K
        def diff_func(k):
            if k < 0.9716: return 0
            # This is a rough approximation for visualization
            return 0.5 * (k**2) * (1 - np.exp(- (k-0.9716)/0.8))

        diff_curve = diff_axes.plot(diff_func, color=YELLOW, x_range=[0.9716, 4])
        kc_line = DashedLine(
            diff_axes.c2p(0.9716, 0), diff_axes.c2p(0.9716, diff_func(4)), color=RED
        )
        kc_label = MathTex("K_c").next_to(kc_line, DOWN)
        
        self.play(Create(diff_axes), Write(x_ax_label), Write(y_ax_label))
        self.play(Create(kc_line), Write(kc_label))
        self.play(Create(diff_curve))
        self.next_slide()
        
    def slide_8_anomalous(self):
        """ANOMALOUS TRANSPORT"""
        self.play(FadeOut(self.wipe_all_but([])))
        title = Title("Anomalous Transport: The Role of Cantori")
        text = Tex(
            r"Immediately above $K_c$, remnants of broken tori (Cantori)",
            r"act as partial barriers, creating 'sticky' regions.",
            r"This leads to sub-diffusion and Lévy flights.",
            font_size=36
        ).center().shift(DOWN*0.5)
        
        self.play(Write(title))
        self.play(Write(text))
        self.next_slide()

    def slide_9_summary(self):
        """SUMMARY"""
        self.play(FadeOut(self.wipe_all_but([])))
        title = Title("Summary and Open Questions")
        summary_points = BulletedList(
            "Standard Map shows a sharp transition from confined to diffusive transport.",
            "Transition is governed by a topological change: destruction of the last KAM torus.",
            "The diffusion coefficient $D(K)$ acts as an effective order parameter.",
            "Future Work: Impact of noise, higher-dimensional systems?",
            font_size=36
        ).center().shift(DOWN*0.5)

        self.play(Write(title))
        self.play(Write(summary_points))
        self.next_slide()

    def slide_10_questions(self):
        """QUESTIONS"""
        self.play(FadeOut(self.wipe_all_but([])))
        q_text = Text("Thank You", font_size=72)
        questions = Text("Questions?", font_size=48).next_to(q_text, DOWN, buff=1)
        self.play(Write(q_text), FadeIn(questions))
        self.next_slide()

    def wipe_all_but(self, mobjects_to_keep):
        """Helper to fade out everything except the specified objects."""
        if not isinstance(mobjects_to_keep, list):
            mobjects_to_keep = [mobjects_to_keep]
        return VGroup(*[mob for mob in self.mobjects if mob not in mobjects_to_keep])