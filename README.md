# Spam Classifier

A web app that tells you whether a text message is spam, built to learn the full ML path: data, training, evaluation, serving, and deployment.

**Status:** Milestone 1 (data exploration). Nothing to serve yet.

Plan and decisions: [docs/DECISIONS.md](docs/DECISIONS.md)

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Get the data

The data is not stored in this repo. Download `sms+spam+collection.zip` from
<https://archive.ics.uci.edu/dataset/228/sms+spam+collection> and unzip it so that the file is at
`data/raw/sms_spam/SMSSpamCollection`.

## Data credit

SMS Spam Collection, licensed CC BY 4.0.
Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84>
