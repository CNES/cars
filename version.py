import os
import subprocess
import sys


def get_vcs():
    subprocess.run(["python", "-m", "setuptools_scm"], check=True)


def set_dist(version):
    meson_project_dist_root = os.getenv("MESON_PROJECT_DIST_ROOT")
    meson_rewrite = os.getenv("MESONREWRITE")

    if not meson_project_dist_root or not meson_rewrite:
        print(
            "Error: Required environment variables MESON_PROJECT_DIST_ROOT or MESONREWRITE are missing."
        )
        sys.exit(1)

    print(meson_project_dist_root)
    rewrite_command = f"{meson_rewrite} --sourcedir {meson_project_dist_root} "
    rewrite_command += f"kwargs set project / version {version}"

    subprocess.run(rewrite_command.split(" "))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <get-vcs | set-dist> [version]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "get-vcs":
        get_vcs()
    elif command == "set-dist":
        if len(sys.argv) < 3:
            print("Error: Missing version argument for set-dist.")
            sys.exit(1)
        set_dist(sys.argv[2])
    else:
        print("Error: Invalid command.")
        sys.exit(1)
