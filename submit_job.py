from pathlib import Path
from typing import Union, Optional
import datetime
import subprocess
import time
import sys

RUNS_DIR = Path(__file__).parent


def sbatch(file: Path) -> int:
    while True:
        try:
            result = subprocess.run(f'sbatch "{file}"', shell=True, stdout=subprocess.PIPE)
            success_msg = "Submitted batch job"
            stdout = result.stdout.decode("utf-8")
            assert success_msg in stdout, result.stderr
            print(stdout)
            job_id = int(stdout.split(" ")[3])
            return job_id
        except AssertionError:
            print("Failed to submit job retry in 1 second")
            time.sleep(1)


def main(
    template: Union[str, Path],
    args: list[tuple[str, Optional[str]]],
    now_str: Optional[str],
    skip_existing: bool,
    yes: bool = False,
):
    template = Path(template)
    if not template.exists():
        raise FileNotFoundError(f"File not found: {template}")

    template_text = template.read_text()
    if "###COMMANDS###" not in template_text:
        raise ValueError(f"Invalid template: {template}")
    if "###JOB_NAME###" not in template_text:
        raise ValueError(f"Invalid template: {template}")
    if "###SBATCH_ARGS###" not in template_text:
        raise ValueError(f"Invalid template (missing ###SBATCH_ARGS### placeholder): {template}")

    sbatch_dir = RUNS_DIR / "logs"
    sbatch_dir.mkdir(exist_ok=True, parents=True)

    template_preview = []
    for line in template_text.splitlines():
        if line.startswith("#SBATCH"):
            template_preview.append(line)
    template_preview = "\n".join(template_preview)

    print(f"Submitting jobs with the following configuration from {template}:")
    print(template_preview)

    to_submit: list[tuple[Path, Optional[str]]] = []
    for arg_str, afterok in args:
        print(f"====== {arg_str}{f' (afterok={afterok})' if afterok else ''} ======")
        arg = Path(arg_str)
        if not arg.exists():
            raise FileNotFoundError(f"File not found: {arg}")
        if skip_existing and now_str is not None:
            if (sbatch_dir / f"{now_str}_{arg.stem}.sbatch").exists():
                print(f"Skipping {arg} because it already exists")
                continue
        print(arg.read_text())
        to_submit.append((arg, afterok))
    print("=========")

    if not yes:
        print(f"Do you want to submit {len(to_submit)} jobs? [y/n]: ", end="")
        answer = input()
        if answer.lower() != "y":
            print("Aborting...")
            return

    now = datetime.datetime.now()
    if now_str is None:
        now_str = now.strftime("%Y%m%d_%H%M")
    for arg, afterok in to_submit:
        job_name = f"{now_str}_{arg.stem}"
        print(f"Submitting {job_name}{f' (afterok={afterok})' if afterok else ''}")
        cmds = (
            arg.read_text()
            .replace("###JOB_NAME###", job_name)
            .replace("###TIMESTAMP###", now_str)
        )
        sbatch_args = f"#SBATCH --dependency=afterok:{afterok}" if afterok else ""
        sbatch_name = sbatch_dir / f"{job_name}.sbatch"
        sbatch_text = (
            template_text.replace("###COMMANDS###", cmds)
            .replace("###JOB_NAME###", job_name)
            .replace("###SBATCH_ARGS###", sbatch_args)
        )
        sbatch_name.write_text(sbatch_text)
        sbatch(sbatch_name)
        time.sleep(0.1)


if __name__ == "__main__":
    argv = sys.argv
    template = Path(argv[1])
    cur = 2
    now_str = None
    skip_existing = False
    yes = False
    while cur < len(argv):
        if argv[cur] == "--time":
            now_str = argv[cur + 1]
            cur += 2
        elif argv[cur] == "--skip":
            skip_existing = True
            cur += 1
        elif argv[cur] == "--yes":
            yes = True
            cur += 1
        else:
            break

    # Each positional arg is a slurm file; an optional --afterok=<jobid> after a file
    # attaches a dependency to that file. The afterok value is passed through to
    # sbatch, so colon-joined multi-dep strings (e.g. --afterok=1234:5678) work.
    file_deps: list[tuple[str, Optional[str]]] = []
    while cur < len(argv):
        path_str = argv[cur]
        cur += 1
        afterok: Optional[str] = None
        if cur < len(argv) and argv[cur].startswith("--afterok="):
            afterok = argv[cur].split("=", 1)[1]
            cur += 1
        file_deps.append((path_str, afterok))

    main(
        template,
        file_deps,
        now_str=now_str,
        skip_existing=skip_existing,
        yes=yes,
    )
