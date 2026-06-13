# Sample Response — T1566 Phishing

**Question:**
Analyze MITRE ATT&CK technique T1566 Phishing. Include attacker behavior, detection ideas, and mitigations.

**Model:** google/gemma-4-12b-it + gemma4-mitre-sft adapter
**Endpoint:** POST /analyze — max_new_tokens: 700

---

**Answer:**

## T1566 — Phishing

**Tactic:** Initial Access (TA0001)

### Attacker Behavior

Phishing is one of the most prevalent initial access vectors. Adversaries craft deceptive electronic communications to trick targets into executing malware, providing credentials, or visiting attacker-controlled infrastructure. Key sub-techniques:

- **T1566.001 — Spearphishing Attachment:** The attacker sends a targeted email with a malicious attachment — commonly a macro-enabled Office document (`.docm`, `.xlsm`), a PDF exploiting a reader vulnerability, or an archive containing a `.js` or `.lnk` dropper. The email is tailored to the target using OSINT to increase credibility (correct name, job title, current project references).
- **T1566.002 — Spearphishing Link:** The email contains a URL leading to a credential harvesting page (lookalike login portal) or a drive-by download site. Adversaries use URL shorteners, typosquatted domains, and HTTPS certificates to appear legitimate.
- **T1566.003 — Spearphishing via Service:** Attacks are delivered through non-email channels — LinkedIn messages, Teams/Slack DMs, SMS, or social media. This bypasses email security gateways entirely.

Threat groups such as APT29 (Cozy Bear), Lazarus Group, and FIN7 have all used spearphishing as their primary initial access technique in documented campaigns.

### Detection Ideas

- **Email gateway analysis:** Scan attachments for macro code, embedded OLE objects, and known exploit signatures. Flag emails with mismatched display names vs. sending domains (header analysis).
- **URL reputation and sandboxing:** Detonate links in a sandbox at click-time. Rewrite URLs through a security proxy to inspect the destination dynamically.
- **Macro execution events:** Log and alert on `winword.exe` or `excel.exe` spawning child processes such as `cmd.exe`, `powershell.exe`, or `wscript.exe` (Sysmon Event ID 1).
- **DNS anomalies:** Newly registered domains and domains with low Alexa/popularity rank used in phishing — correlate DNS queries from email client processes with threat intel feeds.
- **User reporting pipeline:** A low-friction "report phish" button in the mail client remains one of the highest-signal detection sources. Build automated triage on reported emails.

### Mitigations

- **M1049 — Antivirus/Antimalware:** Scan attachments at the gateway and endpoint before opening; use sandboxing for unknown file types.
- **M1054 — Software Configuration:** Disable Office macros by default via Group Policy; only allow macros signed by trusted publishers for users who genuinely need them.
- **M1017 — User Training:** Regular phishing simulation exercises that reflect real adversary TTPs (not just generic "click here" tests) measurably reduce click rates.
- **M1021 — Restrict Web-Based Content:** Block the download of executable file types (`.exe`, `.js`, `.lnk`, `.iso`) at the email gateway and web proxy.
- **M1047 — Audit:** Review email gateway logs for external senders impersonating internal domains; enforce DMARC, DKIM, and SPF to reduce domain spoofing.
- **Multi-Factor Authentication:** MFA on all externally accessible services limits the blast radius of credential phishing even when credentials are successfully harvested.
