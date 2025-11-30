# AI Honeypot Simulator - Demo Script

This script outlines a sequence of actions an attacker might perform against the honeypot, and the expected real-time responses from the AI-driven simulator. This will demonstrate the dynamic content generation capabilities.

---

## Scenario: Reconnaissance and Data Exfiltration Attempt

**Goal**: Simulate an attacker attempting to gain information about the system, access sensitive files, and explore database contents.

**Setup**: 
1.  Ensure the Honeypot Simulator is running (`python main.py`).
2.  Open the Streamlit dashboard in a web browser (e.g., `http://localhost:8501`).
3.  Attacker uses an SSH client (e.g., `ssh admin@<honeypot-ip> -p 2222`).

---

## Attacker Actions & Expected Honeypot Responses

**Attacker IP**: `192.168.1.50` (example)

### Phase 1: Initial Access & System Reconnaissance

| Step | Attacker Action (SSH Terminal)             | Expected Honeypot Response (SSH Terminal)                                                                                             | Expected Dashboard Log Entry                                                                                                       |
| :--- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| **0**| `ssh admin@localhost -p 2222` (password: `admin123`) | `Welcome to Ubuntu... admin@honeypot-server:~$`                                                                                       | `🔗 SSH Connection from 192.168.1.50`                                                                                              |
| **1**| `whoami`                                   | `admin`                                                                                                                               | `💻 Command executed: whoami by 192.168.1.50 (static)`                                                                             |
| **2**| `hostname`                                 | `honeypot-server`                                                                                                                     | `💻 Command executed: hostname by 192.168.1.50 (static)`                                                                           |
| **3**| `ls -l /`                                  | Fake directory listing with permissions, sizes, and timestamps (AI-generated for directory contents)                                  | `💻 Command executed: ls -l / by 192.168.1.50 (ai_generated)`                                                                      |
| **4**| `cd /var/log`                              | `admin@honeypot-server:/var/log$` (prompt changes)                                                                                    | `💻 Command executed: cd /var/log by 192.168.1.50 (processed)`                                                                     |
| **5**| `ls`                                       | Fake listing of log files (e.g., `auth.log`, `syslog`, `apache2/`) (AI-generated for directory contents)                            | `💻 Command executed: ls by 192.168.1.50 (ai_generated)`                                                                           |
| **6**| `tail /var/log/auth.log`                   | Recent fake authentication log entries (TimeGAN/LSTM generated, 10-20 lines)                                                          | `📁 File tail: /var/log/auth.log by 192.168.1.50`<br/>`💻 Command executed: tail /var/log/auth.log by 192.168.1.50 (ai_generated)` |
| **7**| `cat /etc/passwd`                          | Fake `/etc/passwd` content (LLM-generated)                                                                                            | `📁 File read: /etc/passwd by 192.168.1.50`<br/>`💻 Command executed: cat /etc/passwd by 192.168.1.50 (ai_generated)`             |
| **8**| `ps aux`                                   | Fake process list (LLM-generated)                                                                                                     | `💻 Command executed: ps aux by 192.168.1.50 (ai_generated)`                                                                       |
| **9**| `netstat -tuln`                            | Fake network connections and listening ports (LLM-generated, including SSH, HTTP, MySQL)                                              | `💻 Command executed: netstat -tuln by 192.168.1.50 (ai_generated)`                                                                |

### Phase 2: Targeted Data Access

| Step | Attacker Action (SSH Terminal)             | Expected Honeypot Response (SSH Terminal)                                                                                             | Expected Dashboard Log Entry                                                                                                       |
| :--- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| **10**| `cd /home/admin/Documents`                 | `admin@honeypot-server:/home/admin/Documents$`                                                                                        | `💻 Command executed: cd /home/admin/Documents by 192.168.1.50 (processed)`                                                       |
| **11**| `ls -l`                                    | Fake listing: `notes.txt`, `passwords.txt`, `customer_data.csv` (AI-generated for file properties)                                  | `💻 Command executed: ls -l by 192.168.1.50 (ai_generated)`                                                                        |
| **12**| `cat notes.txt`                            | Fake notes content (LLM-generated text about system configurations/sensitive info)                                                    | `📁 File read: notes.txt by 192.168.1.50`<br/>`💻 Command executed: cat notes.txt by 192.168.1.50 (ai_generated)`                 |
| **13**| `cat passwords.txt`                        | Fake password entries (LLM-generated, weak passwords, hashes)                                                                         | `📁 File read: passwords.txt by 192.168.1.50`<br/>`💻 Command executed: cat passwords.txt by 192.168.1.50 (ai_generated)`         |
| **14**| `cat customer_data.csv`                    | Displays synthetic CSV data (CTGAN-generated customer records)                                                                        | `📁 File read: customer_data.csv by 192.168.1.50`<br/>`💻 Command executed: cat customer_data.csv by 192.168.1.50 (ai_generated)` |

### Phase 3: Database Interaction

| Step | Attacker Action (SSH Terminal)             | Expected Honeypot Response (SSH Terminal)                                                                                             | Expected Dashboard Log Entry                                                                                                       |
| :--- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| **15**| `mysql -u root -p` (password: `admin`)     | `Welcome to the MySQL monitor... mysql>`                                                                                              | `💻 Command executed: mysql -u root -p by 192.168.1.50 (processed)`                                                                |
| **16**| `show databases;`                          | Fake list of databases (e.g., `information_schema`, `mysql`, `app_db`, `company_data`)                                            | `💻 Command executed: show databases; by 192.168.1.50 (simulated)`                                                                 |
| **17**| `use app_db;`                              | `Database changed`                                                                                                                    | `💻 Command executed: use app_db; by 192.168.1.50 (simulated)`                                                                      |
| **18**| `show tables;`                             | Fake list of tables (e.g., `customers`, `transactions`, `users`) (CTGAN-based suggestions)                                          | `💻 Command executed: show tables; by 192.168.1.50 (ai_generated)`                                                                 |
| **19**| `SELECT * FROM customers LIMIT 5;`         | Synthetic customer data formatted as MySQL output (CTGAN-generated)                                                                   | `💻 Command executed: SELECT * FROM customers LIMIT 5; by 192.168.1.50 (ai_generated)`                                             |

### Phase 4: Port Scanning (Simulated)

| Step | Attacker Action (SSH Terminal)             | Expected Honeypot Response (SSH Terminal)                                                                                             | Expected Dashboard Log Entry                                                                                                       |
| :--- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| **20**| `nmap localhost`                           | Fake Nmap scan report showing some open ports (e.g., 22, 80, 3306)                                                                    | `💻 Command executed: nmap localhost by 192.168.1.50 (simulated)`                                                                  |

### Phase 5: Exit

| Step | Attacker Action (SSH Terminal)             | Expected Honeypot Response (SSH Terminal)                                                                                             | Expected Dashboard Log Entry                                                                                                       |
| :--- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| **21**| `exit`                                     | `Goodbye!` and SSH connection closes                                                                                                  | `🔗 SSH Connection from 192.168.1.50 (disconnected)`                                                                               |

---

## Demo Walkthrough Notes:

*   **Real-time Interaction**: Emphasize how the dashboard updates instantly with each attacker action and AI generation.
*   **AI Dynamic Content**: Highlight that files and command outputs are not static but are generated on-the-fly, giving a sense of a living system.
*   **Variety of AI Models**: Point out the different AI models (LLM, CTGAN, TimeGAN) at play for various data types.
*   **Customization**: Mention that templates and sample data can be easily customized to create different honeypot personas or data types.

