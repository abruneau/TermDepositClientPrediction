# TermDepositClientPrediction 

## Objective:

Predict if the client will subscribe a term deposit based on data from direct marketing campaigns (phone calls) of a Portuguese banking institution.

## Use Case Description:

Sample data and data set description is available at:

[https://archive.ics.uci.edu/ml/datasets/Bank+Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

Input variables:

1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')
8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

Data will be enriched the `isContactedRecently` flag based on pdays field in dataset. Further it will be filtered based on job to filter out `unknown` description.  

Once data is ready it will be fed to Logistic Regression MLmodel to classify whether client will subscribe to term deposit.

## Pipeline Description:

**Data Enrichment**: Data will be enriched the “isContactedRecently” flag based on pdays field in dataset. Condition would be:

If pDays < 100 then isContactedRecently  = true else isContactedRecently  = false.

**Data Cleansing**: Filter out the records with job as `unknown` and `isContactedRecently` as true.

**LogisticRegression**: LR Classification that determines whether a client will subscribe to term deposit or not based on following features:

* age
* balance 
* duration
* pdays
* job 
* marital
* poutcome

## Demo

Steps:
1. Open a terminal and run setup.sh
2. In your python session run main.py
3. When finished, run cleanup.sh in the terminal

Recommended Session Sizes: 2 CPU, 4 GB RAM