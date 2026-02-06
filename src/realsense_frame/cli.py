import click
from realsense_frame.capture import main as capture_group
from realsense_frame.visualizer import visualize_command


@click.group()
def main():
    """RealSense Frame - capture, visualize, and export RealSense data."""
    pass


# Register all subcommands
# From capture group: capture, list-streams, export-ply
for name, cmd in capture_group.commands.items():
    main.add_command(cmd, name)

# From visualizer
main.add_command(visualize_command)


if __name__ == "__main__":
    main()
