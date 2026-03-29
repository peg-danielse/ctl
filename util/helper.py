import subprocess

def run_remote_command(host: str, user: str, command: str, key_path: str | None = None):
    ssh_cmd = ["ssh"]

    if key_path:
        ssh_cmd += ["-i", key_path]

    # Optional but useful for automation scripts:
    ssh_cmd += ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]

    ssh_cmd += [f"{user}@{host}", command]

    result = subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        check=False
    )

    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

if __name__ == "__main__":
    out = run_remote_command(
        host="192.168.1.50",
        user="myuser",
        command="systemctl status nginx --no-pager",
        key_path="/home/myuser/.ssh/id_ed25519",  # or None if agent/default key
    )
    print(out)