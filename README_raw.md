## Running the pipeline

1.  Activate your virtual environment  
   ```bash
   source venv/bin/activate    # Mac/Linux
   venv\Scripts\activate       # Windows

2.  Install dependencies

    pip install -r requirements.txt

3.  Run all stages

    python main.py

4.  Start MLflow UI to inspect runs

    mlflow ui

Outputs are saved in the artifacts/ directory

---

### My goal is a balance between speed, proffesionalism, and scalability by modular coding
## Created Virtual Enviorment and made a basic structure that i usualy start from to save time ## which is:
# 1. experimentation_notebooks dir to experiment before implementing in components
# 2. src/datascience/__init__.py  - added logger
# 3. src/datascience/utils/common.py - reused past utils functions
# 4. src/datascience/components/
# 5. src/datascience/pipeline/ - inputing the pipelines here
# 6. config/config.yaml - for all configurations

## First off i wanted to turn the purchase-order-items.xlsx to csv because it is easier to handle
# wrote a constant
# wrote data_ingestion configurations
# made 01_data_ingestion notebook 
# wrote a basic configurations manager to get the data_ingestion_config using the constants
# after everything worked in notebook i copy pasted all the codes into place

## Made a EDA NoteBook to explore, ask and asnwer questions
## Findings:
# Dataset is about construction
# Project ID is useless as it doesnt have a single non-null value
# Item Name and Product ID both have 240 null values which makes approximately 7.6% and Tax ID has 65 which is approximately 2%
# Tried to see any value in ID's if somehow they were linked
# Seemed like all of them had the same Account ID except one and it did not have a Item Name and Tax ID was the same across all rows no variance
# For all the ID columns I was not able to explain them and there was no other table to explain them 
# There were 3085 items in SAR currency and 65 USD and even though 1,347,573 USD is the total revenue compared to 151,550,000 SAR it will still be included
# Made a new column Unit Price by deviding (Total Byc/Quantity) to see if large quantities made sense or not and they made sense but there were some purchases with item names and quantity but 0 Byc so they will be dropped because they add no value and they drag the mean down
# Total Bcy and Sub Bcy have the exact same values except for 4 rows so decided on dropping them because they add no value
# Seems like all Item names that had null values had Product ID value as null too, I renamed them to unknown product and saw that they account for 9.9% of total revenue so they are important non-random purchases
# A Unknown Product has the highest revenue with steel, concrete, rubber - materials
# Most expensive by Unit Ptice is capital items or services 
# Most purchased by Quantity were cheap construction materials
# For Quantity the mean was 475.470 and 75% was >=130 huge difference looked like there were values making the mean higher

## started to think about Textual EDA of Item Names I wanted to know which Item Name's where Arabic or English or both, and try to see if i split them based on if they were services, Material, Mixed, or other but had a hard with how to do it time so I used a llm to help me out with what i was thinking.
# Thgought about basic catogrisation and using a llm to validate but I thought it was stupidly expensive and unnecessary
# So to split the rows by services or items I used a small keyword list but it wasn't entirly accurate so I had an idea of an algorithm that makes a larger keyword set covering most of the words in Item Names using llm as api but because llm is expensive and slow so I thought skip the dupes but after starting I relised it's a bit too much and time did not help so decided on going a easier route by asking chatgbt for a keyword list
# there were 1589 catogrised as other so I inspected them and saw that the results were just terible
# then I remembered a thing I did before which is using a smaller model localy from hugging face with pre-defined categories
# So I had to start with preprocessing by implementing everything I concluded about data before and adding text normalisation for both arabic and english

## Preprocessing
# Made preprocessing class and modified ConfigurationManager and datapreprocessingconfig as I went
# Started with a function that drops 'Purchase Order ID', 'Product ID', 'Account ID', 'Tax ID', 'Project ID', 'Item ID', 'Sub Total Bcy' because of reasons discussed above
# Then a function that converts items that were USD to SAR because only 65 rows were in USD and it would make the Analysis & Insights process have more accurate results
# Afterwards I created a function that changes nan Item Names to Uknown Product, because they have high effect/value on Analysis & Insights later as they make 9.9% of total revenue
# Afterwards I created function to drop any rows that have Quantity>0 and Total Bcy==0 because it doesn't make sense and could confuse Analysis & Insights process
# Afterwards I created a function to create a new column which is Unit price because it is a important piece of information and can add value to Analysis & Insights process
# English Text Normalisation by .lower() and trimming spaces for better model clustering and removeing punc
# Arabic Text Normalisation by removing tashkeel and tatweel adn unifying arabic letters like (ا،ه،ي)

## categorising (I used chatgbt for help here because it has been a while since working with hugging face and clustering)
# Before completing preprocessing step I needed to finalise an idea for categorising
# Since I have used hugging face smaller models before I decided to go with a multi-lingual model.
# making the categorisation class and functions was hard I am not the best at clustering and it's been a while so I wanted to go a hybrid approach using 2 models but then thought it was too extreme so I used one hugging face model paired with mlflow to track experiments the first results were shockingly horrible as the keyword rule-based algorithm had better results, the second run I lowered tau to 0.4 and saw more promising results and i will keep all my tests so you can view them when you open my repository
# For the next run decided to add to the seeds.json and also added tf-idf for word count based on model to see what was going wrong but it seems like the model was right about most of the things it labeled other than Others
# I wanted to continue playing with params and adding json keywords but realised this is just a job assesment haha.

## Insights (assuming clusters are accurate)
# First thing I noticed when I saw total expenditure per label is that the highest total spend was on Rebar & Steel at approximately 31% totaling to 48,787,757 SAR, followed by Electrical & Cables which came in second at approximately 17% totaling to 26,887,966 SAR
# Items labeled as other took 3rd place with approximately 13.9% of total expenditure totaling to 21,831,575 SAR
# Afterwards I wanted to see the Unit Price in each category and saw that the most stable category in terms of each item's price was PPE & Safety and it made sense since items like helmets, gloves, safety vests don’t vary wildly on the other hand, the most unstable was Pipes & Fittings --it could mean that it is a result of wrong categorising if I had more time I would defenately investigate it
# By comparing price consistency with clustering confidence, I can see where the model’s grouping is trustworthy or misleading, categories like PPE & Safety and Fasteners have low price_cv, making them reliable on the other hand, clusters like Rebar & Steel and Pipes & Fittings have very high price_cv, meaning items inside differ drastically in price despite moderate confidence, this shows the need for sub-categorisation for engineering materials, while clusters like PPE can stand as-is

## With more time:
# I would have tried unifying the language to use a single language model which will give more accurate results
# I would have tested with more parameters to get better results
# I would have added data_validation as it is essential in any datascience/ml (data related) project to ensure data integrity
# I would have tried different categorising techniques like the first one I thought of which was creating a keyword dictionary using llm api on data set to extract all the words and skip dupes then use a rule based algorithm as it will be faster and cheaper in long run



