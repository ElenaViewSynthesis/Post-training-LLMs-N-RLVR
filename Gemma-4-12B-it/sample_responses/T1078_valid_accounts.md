# Sample Response — T1078 Valid Accounts

**Question:**
Analyze MITRE ATT&CK technique T1078 Valid Accounts. Include attacker behavior, detection ideas, and mitigations.

**Model:** google/gemma-4-12b-it + gemma4-mitre-sft adapter
**Endpoint:** POST /analyze — max_new_tokens: 700

---

**Answer:**

## T1078 — Valid Accounts

**Tactic:** Defense Evasion, Persistence, Privilege Escalation, Initial Access (TA0005, TA0003, TA0004, TA0001)

### Attacker Behavior

Using valid credentials is one of the stealthiest techniques available to an adversary because it blends into normal authentication traffic. Accounts are obtained through credential dumping, phishing, purchasing on criminal markets, or brute-force attacks on exposed services. Sub-techniques:

- **T1078.001 — Default Accounts:** Attackers exploit unchanged vendor-default credentials on network devices, IoT hardware, databases (e.g., `admin/admin`, `sa/` on MSSQL), and cloud service accounts. Widely exploited in OT/ICS environments.
- **T1078.002 — Domain Accounts:** Compromised Active Directory user or service accounts allow lateral movement across the domain. Service accounts often have elevated privileges and weak rotation policies, making them high-value targets.
- **T1078.003 — Local Accounts:** Built-in local Administrator accounts (especially with the same password across a fleet) allow adversaries to pivot between machines without triggering Kerberos-level alerts.
- **T1078.004 — Cloud Accounts:** Compromised cloud console credentials (AWS IAM, Azure AD, GCP) give access to infrastructure, object storage, and secrets managers. Often achieved via credential stuffing against SSO portals.

APT groups commonly use valid accounts in the persistence and lateral movement phases — the account already exists, so no new artifacts are created.

### Detection Ideas

- **Impossible travel / geolocation anomalies:** Authentication from two geographically distant locations within a short time window is a strong signal of credential compromise.
- **Off-hours login alerts:** Flag successful authentications outside established working hours for high-privilege accounts.
- **Logon type analysis (Windows Event 4624):** Logon Type 3 (network) or Type 10 (RemoteInteractive) for service accounts that normally never authenticate interactively is highly suspicious.
- **Failed authentication spikes (Event 4625):** A burst of failed logins before a successful one indicates brute-force or credential stuffing.
- **Service account interactive logons:** Service accounts should never produce interactive sessions. Alert immediately on any Event ID 4624 with Type 2 or 10 for these accounts.
- **Cloud: CloudTrail / Audit Logs:** Alert on API calls from unusual IP addresses, new regions, or at unusual times — especially `AssumeRole`, `GetSecretValue`, or `ListBuckets` from accounts that don't normally call them.

### Mitigations

- **M1026 — Privileged Account Management:** Enforce least-privilege; separate service accounts from human accounts; rotate service account passwords regularly and store in a secrets manager.
- **M1032 — Multi-Factor Authentication:** Apply MFA to all externally accessible services and privileged accounts. MFA is the single most effective control against credential-based attacks.
- **M1027 — Password Policies:** Enforce strong, unique passwords; integrate with HIBP or similar breach databases to block known-compromised passwords on set/change.
- **M1036 — Account Use Policies:** Define and enforce expected behavior for each account type (allowed hours, source IPs, logon types) and alert on deviations.
- **M1017 — User Training:** Educate users on credential phishing, MFA fatigue attacks, and secure credential hygiene.
- **Disable Default Credentials:** Immediately change or disable default credentials on all network devices, databases, and cloud service accounts during provisioning.
