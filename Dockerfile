FROM python:3.6-slim
COPY /app /app
WORKDIR /app
RUN mkdir ~/.streamlit
RUN cp .streamlit/credentials.toml ~/.streamlit/credentials.toml
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]