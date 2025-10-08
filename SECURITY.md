# 🔐 Security Configuration Guide

## Overview

This document explains how to securely configure the Face Recognition System using environment variables to protect sensitive information like API keys, connection strings, and credentials.

## 🗂️ Environment Files

### `.env` (Production Configuration)
- **Contains**: Real production secrets and configuration
- **Location**: `/home/ubuntu24/ids/.env`
- **Security**: ⚠️ **NEVER COMMIT TO GIT** - Protected by `.gitignore`

### `.env.example` (Template)
- **Contains**: Example configuration with placeholder values
- **Location**: `/home/ubuntu24/ids/.env.example`  
- **Security**: ✅ Safe to commit - No real secrets

## 🔑 Critical Secret Variables

### Azure Storage Secrets
```bash
# HIGH SENSITIVITY - Azure Storage Account Key
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=sakarguard;...

# Connection details
AZURE_STORAGE_ACCOUNT_NAME=sakarguard
AZURE_CONTAINER_NAME=sr001
```

### API Security
```bash
# Change from default in production
API_SECRET_KEY=your-secret-api-key-here
```

### Email Credentials (Optional)
```bash
# Email notification credentials
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 🛡️ Security Best Practices

### 1. Environment Variable Protection
- ✅ Use `.env` files for local development
- ✅ Use system environment variables in production
- ✅ Never hardcode secrets in source code
- ❌ Never commit `.env` files to version control

### 2. Access Control
```bash
# Set proper file permissions
chmod 600 .env          # Only owner can read/write
chmod 644 .env.example  # Example file can be read by all
```

### 3. Production Deployment
```bash
# Option 1: System environment variables
export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"

# Option 2: Docker secrets
docker run -e AZURE_STORAGE_CONNECTION_STRING="..." your-app

# Option 3: Kubernetes secrets
kubectl create secret generic face-recognition-secrets \
  --from-literal=azure-connection-string="..."
```

## 🔧 Configuration Management

### Loading Configuration
The system automatically loads configuration in this order:
1. **System environment variables** (highest priority)
2. **`.env` file** in project root
3. **Default values** (fallback)

### Configuration Validation
```bash
# Test configuration
python config.py

# Expected output:
✅ Configuration: VALID
⚠️  Warnings: Default API secret key detected
```

### Environment-Specific Configurations

#### Development
```bash
# .env.development
RECOGNITION_THRESHOLD=0.5
LOG_LEVEL=DEBUG
```

#### Production
```bash
# .env.production  
RECOGNITION_THRESHOLD=0.7
LOG_LEVEL=INFO
API_SECRET_KEY=production-secret-key
```

## 🚨 Security Checklist

### Before Production Deployment
- [ ] Change default API secret key
- [ ] Verify Azure connection string is correct
- [ ] Set appropriate file permissions on `.env`
- [ ] Configure secure SMTP credentials
- [ ] Test configuration validation
- [ ] Ensure `.env` is in `.gitignore`
- [ ] Review all default values
- [ ] Enable logging for security events

### Regular Security Maintenance
- [ ] Rotate Azure storage keys periodically
- [ ] Update API secret keys
- [ ] Review access logs
- [ ] Monitor for unauthorized access attempts
- [ ] Update dependencies for security patches

## 🔍 Configuration Variables Reference

### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure blob storage connection | `DefaultEndpointsProtocol=https;...` |
| `AZURE_CONTAINER_NAME` | Container name | `sr001` |
| `AZURE_BLOB_NAME` | Blob path | `authorised/authorized_persons.json` |

### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `RECOGNITION_THRESHOLD` | `0.6` | Face recognition confidence |
| `DETECTION_SIZE_WIDTH` | `640` | Detection image width |
| `API_SECRET_KEY` | `your-secret-api-key-here` | API authentication |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Missing Azure Connection String
```
Error: AZURE_STORAGE_CONNECTION_STRING not found
Solution: Add the connection string to .env file
```

#### 2. Invalid Configuration
```
Error: Configuration validation failed
Solution: Run `python config.py` to check issues
```

#### 3. Permission Denied
```
Error: Cannot read .env file
Solution: Check file permissions with `ls -la .env`
```

## 📞 Support

For security-related issues:
1. **Check configuration**: `python config.py`
2. **Validate environment**: Check `.env` file exists and has correct permissions
3. **Test Azure connection**: Run `python azure_integration.py`
4. **Review logs**: Check `face_recognition.log` for errors

## 🔒 Emergency Procedures

### If Secrets Are Compromised
1. **Immediately rotate** Azure storage keys
2. **Generate new** API secret keys  
3. **Update** all environment configurations
4. **Review** access logs for unauthorized usage
5. **Test** system functionality after updates

### Backup and Recovery
1. **Backup** `.env.example` with updated templates
2. **Document** all configuration changes
3. **Test** restore procedures in development
4. **Maintain** separate production/development configs