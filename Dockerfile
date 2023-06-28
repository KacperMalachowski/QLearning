FROM python:3.10

ENV NON_HUMAN=true

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "train.py" ]