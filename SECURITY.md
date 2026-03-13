# Security Policy

## Reporting a Vulnerability

MnemeFusion is a database library that may store sensitive conversational data. We take security seriously.

If you discover a security vulnerability, please report it privately:

- **Email**: [gkanellopoulos@protonmail.com](mailto:gkanellopoulos@protonmail.com)
- **Subject**: `[SECURITY] MnemeFusion: <brief description>`

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation within 7 days for critical issues.

## Scope

The following are in scope for security reports:

- Data leakage between namespaces
- Unauthorized access to `.mfdb` file contents
- Memory safety issues in the Rust core
- Injection vulnerabilities in query processing
- Unsafe deserialization of stored data

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Considerations

- `.mfdb` files are not encrypted at rest. Use filesystem-level encryption for sensitive data.
- Entity extraction uses local GGUF models — no data is sent to external services.
- The Python bindings use PyO3 with safe Rust interop; `unsafe` blocks are limited to FFI boundaries (llama-cpp, usearch).
