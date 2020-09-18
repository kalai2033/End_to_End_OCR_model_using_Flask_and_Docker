#FROM python:3.7.4
FROM intelpython/intelpython3_core:2019.4
LABEL maintainer="GenieBilal"
 
WORKDIR /app/
 
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y libgtk2.0-dev
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
 
COPY wsgi.py __init__.py /app/
COPY mergedModel /app/
 
EXPOSE 5000
 
#ENTRYPOINT python ./mergedModels.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app", "--timeout", "90"]
 
#save this file as 'Dockerfile'
