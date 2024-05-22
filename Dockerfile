FROM python:3.11-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install  --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]