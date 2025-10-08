# Deployment Guide

## Production Deployment

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/prajwalmapari/ids.git
cd ids

# Set up Python environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

Required environment variables:
- `AZURE_CONNECTION_STRING`: Your Azure Storage connection string
- `FLASK_SECRET_KEY`: Random secret key for Flask sessions
- Additional API keys as needed

### 3. Azure Storage Setup

1. Create Azure Storage Account
2. Create container named `sakarguard`
3. Upload `authorized_persons.json` with format:
```json
{
  "persons": [
    {
      "id": "person_1234567890",
      "name": "John Doe",
      "encoding": [...],
      "metadata": {...}
    }
  ]
}
```

### 4. Testing

```bash
# Test configuration
python config.py

# Test Azure connection
python azure_integration.py

# Test face recognition
python main.py --mode image --input test_image.jpg
```

### 5. Production Considerations

- Use strong, unique `FLASK_SECRET_KEY`
- Regularly rotate Azure connection strings
- Monitor face recognition confidence thresholds
- Implement proper logging and monitoring
- Use HTTPS in production
- Consider rate limiting for API endpoints

## Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "main.py", "--mode", "api"]
```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| AZURE_CONNECTION_STRING | Azure Storage connection string | Yes |
| FLASK_SECRET_KEY | Flask session secret key | Yes |
| RECOGNITION_THRESHOLD | Face recognition confidence threshold | No (0.6) |
| MAX_FACES | Maximum faces to detect | No (10) |
| DEBUG_MODE | Enable debug logging | No (False) |