# Sample Response — T1059 Command and Scripting Interpreter

**Question:**
Analyze MITRE ATT&CK technique T1059 Command and Scripting Interpreter. Include attacker behavior, detection ideas, and mitigations.

**Model:** google/gemma-4-12b-it + gemma4-mitre-sft adapter
**Endpoint:** POST /analyze — max_new_tokens: 700

---

**Answer:**

## T1059 — Command and Scripting Interpreter

**Tactic:** Execution (TA0002)

### Attacker Behavior

Adversaries abuse command and script interpreters already present on the target system to execute malicious code without dropping a standalone binary. Common sub-techniques include:

- **T1059.001 — PowerShell:** Attackers invoke encoded commands (`-EncodedCommand`), bypass execution policy (`-ExecutionPolicy Bypass`), or download and execute payloads entirely in memory using `IEX (New-Object Net.WebClient).DownloadString(...)`. PowerShell's deep OS integration makes it a high-value living-off-the-land target.
- **T1059.003 — Windows Command Shell (cmd.exe):** Used for batch scripting, chaining commands with `&&` or `|`, and spawning child processes. Frequently seen in ransomware pre-encryption stages (e.g., `vssadmin delete shadows`).
- **T1059.006 — Python:** Deployed where Python is available, especially on Linux servers and developer machines. Allows socket-based reverse shells in a few lines.
- **T1059.007 — JavaScript/JScript:** Executed via `wscript.exe` or `cscript.exe`; used in phishing lure delivery (`.js` attachments) and initial access stages.

Attackers favor these interpreters because they are signed, trusted, often whitelisted by AV, and capable of operating entirely in memory.

### Detection Ideas

- **Script block logging (PowerShell):** Enable `ScriptBlockLogging` and `ModuleLogging` via Group Policy. Alert on encoded commands, `DownloadString`, `Invoke-Expression`, and `AmsiUtils` bypass patterns.
- **Process creation events (Sysmon Event ID 1):** Watch for `cmd.exe`, `powershell.exe`, `wscript.exe`, or `cscript.exe` spawned from unusual parents (e.g., `winword.exe`, `excel.exe`, `outlook.exe`).
- **Network connections from interpreter processes:** A PowerShell or Python process opening an outbound TCP connection is a strong signal — correlate with Sysmon Event ID 3.
- **Command-line length anomalies:** Encoded PowerShell commands are long. Alert on `-EncodedCommand` arguments exceeding a length threshold.
- **AMSI telemetry:** Endpoint security products that integrate AMSI will log malicious script content before execution.

### Mitigations

- **M1038 — Execution Prevention:** Use AppLocker or Windows Defender Application Control (WDAC) to restrict which users and paths can execute `powershell.exe`, `wscript.exe`, and `cscript.exe`.
- **M1042 — Disable or Remove Feature or Program:** Remove scripting runtimes (VBScript, JScript) from systems where they serve no operational purpose.
- **M1026 — Privileged Account Management:** Restrict which accounts can run interpreters with elevated privileges; limit PowerShell remoting to administrators.
- **M1049 — Antivirus/Antimalware:** Ensure AMSI-integrated AV is deployed and up to date; AMSI provides real-time visibility into script content.
- **Constrained Language Mode (PowerShell):** Enforce CLM on endpoints to limit what PowerShell can call, blocking most in-memory execution techniques.
