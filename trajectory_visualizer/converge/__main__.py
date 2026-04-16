"""Launch the Converge Gradio app via `python -m trajectory_visualizer.converge`."""

from .app import build_ui


def main():
    app = build_ui()
    app.launch()


if __name__ == "__main__":
    main()
