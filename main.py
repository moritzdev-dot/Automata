from cli import main
import subprocess


if __name__ == "__main__":
    main()
    subprocess.run("rm *.png", shell=True, check=True)