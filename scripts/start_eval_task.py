"""Register and start a Windows Scheduled Task for LoCoMo eval."""
import subprocess
import time

task_name = "MnemeFusionEvalKG"
bat_path = r"C:\Users\georg\projects\mnemefusion\scripts\run_eval_kg.bat"

subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"], capture_output=True)

result = subprocess.run(
    ["schtasks", "/create", "/tn", task_name, "/tr", bat_path,
     "/sc", "once", "/st", "00:00", "/rl", "highest", "/f"],
    capture_output=True, text=True
)
print("Create:", result.stdout.strip(), result.stderr.strip())

result = subprocess.run(
    ["schtasks", "/run", "/tn", task_name],
    capture_output=True, text=True
)
print("Run:", result.stdout.strip(), result.stderr.strip())

time.sleep(3)
result = subprocess.run(
    ["schtasks", "/query", "/tn", task_name, "/fo", "list"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines():
    if "Status" in line or "Last Run" in line:
        print(line.strip())
