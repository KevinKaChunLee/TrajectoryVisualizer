"""Entry point for `python -m trajectory_visualizer.insight`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Insight")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    try:
        from .insight import build_ui, APP_CSS
    except ImportError:
        print(
            "Error: Insight dependencies are not installed.\n"
            'Install them with: pip install -r requirements.txt',
            file=sys.stderr,
        )
        sys.exit(1)

    app = build_ui()
    app.launch(server_port=args.port, share=args.share, css=APP_CSS)


if __name__ == "__main__":
    main()
