# Security Policy

## 🔒 Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, email us at: **security@frametrain.ai**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### Response Time

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity

### Severity Levels

- 🔴 **Critical**: Remote code execution, authentication bypass
- 🟠 **High**: Data breach, privilege escalation
- 🟡 **Medium**: Cross-site scripting, information disclosure
- 🟢 **Low**: Minor issues with limited impact

## 🛡️ Security Measures

### Authentication & Authorization

- ✅ JWT tokens with secure signing
- ✅ API keys with database validation
- ✅ bcrypt password hashing
- ✅ HTTPS-only in production

### Data Protection

- ✅ All training data stays local (never sent to servers)
- ✅ API keys stored securely in database
- ✅ No telemetry or tracking without consent
- ✅ GDPR compliant

### Payment Security

- ✅ Stripe Payment Processing (PCI DSS compliant)
- ✅ No credit card data stored on our servers
- ✅ Webhook signature verification

### Infrastructure

- ✅ Environment variables for secrets
- ✅ No secrets in version control
- ✅ Regular dependency updates
- ✅ Automated security scans

## 🔐 Best Practices for Users

### API Keys

```bash
# ❌ BAD: Don't commit API keys
FRAMETRAIN_KEY=abc123

# ✅ GOOD: Use environment variables
export FRAMETRAIN_KEY=your_key_here
```

### .env Files

```bash
# Always add to .gitignore
.env
.env.local
.env.*.local
```

### Database Credentials

```bash
# ❌ BAD: Default credentials
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/db"

# ✅ GOOD: Strong, unique passwords
DATABASE_URL="postgresql://user:Xy9$mK2!pQ7@localhost:5432/db"
```

## 📦 Dependencies

We regularly update dependencies to patch security vulnerabilities.

### Automated Checks

- GitHub Dependabot alerts enabled
- npm audit in CI/CD pipeline
- Rust cargo audit

### Manual Reviews

- Monthly security audit of dependencies
- Review of all third-party libraries before use

## 🚨 Known Issues

Currently no known security issues.

Check [GitHub Security Advisories](https://github.com/YourUsername/FrameTrain/security/advisories) for updates.

## 📜 License & Terms

This project is licensed under BSL 1.1. See [LICENSE](./LICENSE).

By using FrameTrain, you agree to:
- Not use it for malicious purposes
- Comply with all applicable laws
- Report security vulnerabilities responsibly

## 🔄 Updates

- **Last Updated**: 2024-01-15
- **Next Review**: 2024-04-15

## 📞 Contact

- Security Issues: security@frametrain.ai
- General Support: support@frametrain.ai
- Website: https://frametrain.ai

---

**Thank you for helping keep FrameTrain secure!** 🙏
