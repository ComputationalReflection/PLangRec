FROM python:3.11-slim-bullseye
COPY web-api /app/web-api
COPY common /app/common
WORKDIR /app/web-api
COPY web-api-requirements.txt .
RUN pip install --no-cache-dir -r web-api-requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]