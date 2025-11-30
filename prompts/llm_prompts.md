# LLM Prompt Templates for AI Honeypot Simulator

This document outlines example prompts and templates used by the GPT-2 based `TextContentGenerator` to dynamically create realistic fake content for the honeypot environment.

---

## 1. Generating Fake Text Files (`notes.txt`, `configs`, `/var/log/*.log`)

The `generate_file_content(filename: str, content_type: str = "general")` method uses these templates, selecting based on file extension or a general prompt.

### Template Category: `file_contents` (for .txt, .md, .readme)

**Purpose**: To generate general-purpose text content that appears to be user-generated notes, documents, or informational files.

**Example Prompts**:

*   "Important notes about the system: Today we upgraded the firewall rules to block"
*   "Meeting notes from yesterday: Discussed the Q3 financial report and decided to invest in"
*   "TODO list for this week: Need to deploy the new application version, then configure the"
*   "Customer information: List of premium customers and their preferred services includes"
*   "Project ideas for next quarter: We should explore integrating AI capabilities into our"
*   "Personal diary entry: Felt really tired after working on the new feature, but it's almost"

**Generated Content Example (for `notes.txt` from "Important notes about the system:")**:
```
Important notes about the system: Today we upgraded the firewall rules to block all incoming traffic on port 22 except from trusted IPs. We also implemented a new monitoring solution to track unusual activity. The database server was patched for a critical vulnerability last night. Remember to update the documentation with these changes by end of week.
```

### Template Category: `config_files` (for .conf, .config, .cfg)

**Purpose**: To generate fake configuration file content for various services.

**Example Prompts**:

*   "# Configuration file for web server
ServerRoot /etc/apache2
Listen 80
LoadModule"
*   "# Database configuration
host=localhost
port=3306
user=admin
password="
*   "# Network configuration
interface eth0
ip_address=192.168.1."
*   "# System log configuration
log_level=INFO
log_file=/var/log/"
*   "# SSH Daemon configuration
Port 22
PermitRootLogin no
PasswordAuthentication yes"

**Generated Content Example (for `/etc/apache2/apache2.conf` from "# Configuration file for web server")**:
```
# Configuration file for web server
ServerRoot /etc/apache2
Listen 80
LoadModule dir_module modules/mod_dir.so
<Directory />
    AllowOverride None
    Require all denied
</Directory>

User ${APACHE_RUN_USER}
Group ${APACHE_RUN_GROUP}
```

### Template Category: `log_entries` (for `/var/log/*.log` files when accessed via `cat`)

**Purpose**: To generate individual realistic-looking log entries. The `LogSequenceGenerator` is preferred for time-series logs, but this can provide quick snippets.

**Example Prompts**:

*   "[INFO] User login attempt from IP 192.168.1.100 for user 'john.doe' at"
*   "[ERROR] Failed to connect to database 'app_db' on host 'localhost': Connection refused."
*   "[WARN] High CPU usage detected on server 'webserver01': Current usage is 85%."
*   "[DEBUG] Processing request for file '/var/www/html/index.php' from client"
*   "[CRITICAL] System halted due to unrecoverable kernel error. Reboot initiated."

**Generated Content Example**:
```
2024-09-04 10:35:12 [INFO] User login attempt from IP 192.168.1.100 for user 'john.doe' at 2024-09-04 10:35:12. Authentication successful.
```

---

## 2. Generating Realistic Command Outputs (e.g., `fake ls` directory contents, `fake /etc/passwd`)

The `generate_command_output(command: str)` method dynamically selects prompts based on the command.

### Command-Specific Prompts:

*   **`ps aux` / `top`**:
    *   **Prompt**: "Process list shows the following active services:
"
    *   **Generated Example**:
        ```
        Process list shows the following active services:
        USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
        root         1  0.0  0.4  77616  8784 ?        Ss   Sep01   0:02 /sbin/init
        root         2  0.0  0.0      0     0 ?        S    Sep01   0:00 [kthreadd]
        admin     1234  0.1  2.1 123456 43210 pts/0    S    15:30   0:05 python3 honeypot.py
        www-data  5678  0.0  1.2  98765 24680 ?        S    15:42   0:00 /usr/sbin/apache2 -k start
        ```

*   **`netstat -an` / `ss -tuln`**:
    *   **Prompt**: "Network connections currently established:
"
    *   **Generated Example**:
        ```
        Network connections currently established:
        Proto Recv-Q Send-Q Local Address           Foreign Address         State      
        tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN     
        tcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN     
        tcp        0      0 127.0.0.1:3306          0.0.0.0:*               LISTEN     
        tcp        0      0 192.168.1.100:22        192.168.1.50:54321      ESTABLISHED
        udp        0      0 0.0.0.0:68              0.0.0.0:*                          
        ```

*   **`ls -l /home/admin`**:
    *   **Prompt**: "Directory listing contains the following files:
"
    *   **Generated Example**:
        ```
        Directory listing contains the following files:
        drwxr-xr-x 2 admin admin   4096 Sep  3 14:22 Documents
        drwxr-xr-x 2 admin admin   4096 Aug 28 09:15 Downloads
        -rw-r--r-- 1 admin admin  15360 Sep  4 10:05 .bash_history
        drwxr-xr-x 2 admin admin   4096 Jun 15 11:00 .ssh
        ```

*   **`cat /etc/passwd`**:
    *   **Prompt (indirectly via `generate_file_content`)**: "Content for file /etc/passwd:
"
    *   **Generated Example**:
        ```
        Content for file /etc/passwd:
        root:x:0:0:root:/root:/bin/bash
        daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
        bin:x:2:2:bin:/bin:/usr/sbin/nologin
        sys:x:3:3:sys:/dev:/usr/sbin/nologin
        sync:x:4:65534:sync:/bin:/bin/sync
        admin:x:1000:1000:Admin User,,,:/home/admin:/bin/bash
        guest:x:1001:1001:Guest Account,,,:/home/guest:/bin/bash
        ```

*   **General/Unknown Commands**:
    *   **Prompt**: "Output for command '{command}':
"
    *   **Generated Example (for `whois example.com`)**:
        ```
        Output for command: whois example.com
        Domain Name: EXAMPLE.COM
        Registry Domain ID: 2987340_DOMAIN_COM-VRSN
        Registrar WHOIS Server: whois.iana.org
        Registrar URL: http://www.iana.org
        Updated Date: 2023-01-01T12:00:00Z
        Creation Date: 1999-09-01T00:00:00Z
        ```

---

## 3. CTGAN/TVAE Prompts for Tabular Data

The `TabularDataGenerator` uses pre-defined schemas and sample data to train its models. Prompts are implicitly defined by the structure and content of the training data.

### Example Schemas (from `_define_common_schemas` in `tabular_generator.py`):

*   **`customers`**:
    *   **Columns**: `customer_id`, `name`, `email`, `phone`, `age`, `city`, `country`, `registration_date`
    *   **Sample Data**: Generated internally to mimic realistic customer records.

*   **`transactions`**:
    *   **Columns**: `transaction_id`, `customer_id`, `amount`, `currency`, `status`, `timestamp`, `merchant`
    *   **Sample Data**: Generated internally to mimic realistic financial transactions.

*   **`users`**:
    *   **Columns**: `user_id`, `username`, `email`, `role`, `last_login`, `status`, `department`
    *   **Sample Data**: Generated internally to mimic corporate user directories.

*   **`logs`**:
    *   **Columns**: `log_id`, `timestamp`, `level`, `service`, `message`, `ip_address`, `user_id`
    *   **Sample Data**: Generated internally to mimic system log entries (distinct from time-series log sequences).

**Implicit "Prompt"**: "Generate 100 synthetic rows that resemble the structure and statistical properties of the `customers` table."

**Generated Content Example (for `customers.csv`)**:
```csv
customer_id,name,email,phone,age,city,country,registration_date
1,Jane Doe,jane.doe@example.com,+1-555-123-4567,35,New York,USA,2024-01-15
2,John Smith,john.smith@example.com,+1-555-987-6543,42,Los Angeles,USA,2023-11-20
3,Alice Johnson,alice.j@example.com,+44-20-7123-4567,28,London,UK,2024-03-01
4,Robert Brown,r.brown@example.com,+61-2-9876-5432,50,Sydney,Australia,2022-07-10
5,Maria Garcia,maria.g@example.com,+34-91-123-4567,31,Madrid,Spain,2024-05-22
```

---

## 4. TimeGAN/LSTM for Time-Series Logs

The `LogSequenceGenerator` focuses on generating sequences of log events over time.

### Example Prompts/Configuration:

*   **`train_timegan_model(model_name: str = "system_logs", num_sequences: int = 200, epochs: int = 50)`**:
    *   **Purpose**: Train a model to understand the temporal patterns and features of system logs. The "prompt" here is the historical log data used for training.

*   **`generate_log_sequence(model_name: str = "system_logs", num_hours: int = 24)`**:
    *   **Purpose**: Generate a sequence of log entries for a specified duration (e.g., 24 hours), maintaining realistic temporal characteristics.
    *   **Implicit "Prompt"**: "Generate a 24-hour sequence of system log events consistent with the 'system_logs' model's learned patterns."

*   **`generate_auth_log(num_entries: int = 20)` / `generate_syslog(num_entries: int = 30)`**:
    *   **Purpose**: Generate specific types of log entries with pre-defined formats and randomized data. These use internal static templates combined with randomization.

**Generated Content Example (for `auth.log` via `tail`)**:
```
Sep  4 10:30:05 honeypot sshd[1234]: Accepted password for admin from 192.168.1.50 port 54321 ssh2
Sep  4 10:30:15 honeypot sshd[1235]: authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=192.168.1.51 user=root
Sep  4 10:31:01 honeypot sshd[1236]: Accepted password for user22 from 192.168.1.55 port 42134 ssh2
Sep  4 10:32:03 honeypot sshd[1237]: Accepted password for admin from 192.168.1.50 port 54322 ssh2
```



