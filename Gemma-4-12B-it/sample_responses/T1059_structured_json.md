# Sample Response — T1059 Structured JSON

**Model:** google/gemma-4-12b-it + gemma4-mitre-sft adapter
**Endpoint:** POST /analyze

---

## Prompt options

**Option A — JSON summary request:**
```json
{
  "question": "Return a structured JSON summary for ATT&CK technique Command and Scripting Interpreter (T1059). Include attacker behavior, tactics, platforms, detection ideas, data sources, mitigations, and defender recommendations.",
  "max_new_tokens": 2000
}
```

**Option B — Defensive analysis with sections:**
```json
{
  "question": "Create a defensive security analysis for MITRE ATT&CK T1059 Command and Scripting Interpreter. Use sections: technique summary, attacker behavior, common abuse patterns, detection logic, useful telemetry, mitigations, and defensive test cases.",
  "max_new_tokens": 2000
}
```

---

## Answer

```json
{
  "tactic": "Execution",
  "technique_id": "T1059",
  "technique_name": "Command and Scripting Interpreter",
  "platforms": [
    "ESXi",
    "Linux",
    "macOS",
    "Network Devices",
    "Windows"
  ],
  "summary": "Adversaries may abuse command and script interpreters to execute commands, scripts, and binaries. Command and script interpreters are programs that execute commands and scripts provided to them by users or other programs. These interpreters are often included as part of a larger application or operating system. Examples include the Windows command interpreter (cmd.exe), macOS and Linux shells (sh, bash, zsh), PowerShell, and the script utility on Windows. Command and script interpreters are also present on network devices, such as the Network Device CLI on Cisco IOS. Adversaries may abuse these interpreters to execute malicious payloads, perform discovery, and carry out other actions on a victim host. Because they are present on a system by default, adversaries can use them to execute commands without introducing additional files.",
  "behavior": [
    "Execution",
    "Discovery",
    "Persistence",
    "Privilege Escalation"
  ],
  "common_abuse_patterns": [
    "PowerShell encoded commands (-EncodedCommand) to bypass logging and AV",
    "cmd.exe spawned from Office processes (winword.exe, excel.exe) as macro payload delivery",
    "In-memory execution via IEX (Invoke-Expression) and DownloadString to avoid disk artifacts",
    "Bash/Python reverse shells on Linux servers and developer machines",
    "wscript.exe / cscript.exe executing .js or .vbs droppers from phishing attachments",
    "Network device CLI abuse for persistence and lateral movement in OT/ICS environments"
  ],
  "detection_logic": [
    "Alert on cmd.exe or powershell.exe spawned from Office parent processes",
    "Flag -EncodedCommand arguments in PowerShell process creation events",
    "Monitor wscript.exe and cscript.exe spawning child processes",
    "Detect interpreter processes (python, bash) opening outbound TCP connections",
    "Alert on command-line arguments containing DownloadString, IEX, or Invoke-Expression",
    "Use PowerShell ScriptBlockLogging and ModuleLogging to capture script content before execution"
  ],
  "useful_telemetry": [
    "Sysmon Event ID 1 — Process creation (parent/child relationships)",
    "Sysmon Event ID 3 — Network connection from interpreter processes",
    "Windows Event ID 4688 — Process creation with command-line logging enabled",
    "PowerShell Event ID 4104 — Script block logging",
    "EDR process tree and command-line telemetry",
    "AMSI telemetry for real-time script content inspection"
  ],
  "data_sources": [
    "Process: Process Creation",
    "Command: Command Execution",
    "Script: Script Execution",
    "Network Traffic: Network Connection Creation",
    "Module: Module Load"
  ],
  "recommended_mitigation": [
    "Execution Prevention — AppLocker or WDAC to restrict interpreter execution by path and publisher",
    "Disable or Remove Feature or Program — remove VBScript and JScript runtimes where not needed",
    "Privileged Account Management — restrict which accounts can run interpreters with elevated privileges",
    "User Account Management — limit PowerShell remoting to administrator accounts only",
    "Restrict File and Directory Permissions — prevent write access to directories from which scripts execute",
    "Code Signing — enforce PowerShell Constrained Language Mode and require signed scripts"
  ],
  "defensive_test_cases": [
    "Run Atomic Red Team T1059.001 — verify PowerShell ScriptBlockLogging captures encoded commands",
    "Simulate cmd.exe spawned from winword.exe — confirm EDR/SIEM alert fires within 60 seconds",
    "Execute a Python reverse shell on a Linux endpoint — verify network connection alert triggers",
    "Attempt wscript.exe execution of a .js payload — confirm AppLocker blocks and logs the attempt",
    "Run IEX DownloadString in PowerShell — verify AMSI detects and blocks the payload"
  ],
  "recommended_tactics": [
    "Execution"
  ],
  "recommended_technique_id": "T1059"
}
```
