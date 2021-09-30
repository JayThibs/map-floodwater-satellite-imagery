streamlit-app:
	docker build -t streamlit-app:latest .f Dockerfile
	docker run -p 8501:8501 streamlit-app:latest