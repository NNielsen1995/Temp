import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime


class BankingData:
    """
    Pipeline for processing banking datasets using pandas
    Handles data extraction, transformation, and analysis
    
    Attributes:
        data_path (str): Path to raw input data
    """
    
    def __init__(self, data_path: str = "https://raw.githubusercontent.com/NNielsen1995/Temp/refs/heads/main"):
        """
        Initialize the banking data
        Args:
            data_path (str): Github raw url base path
                           Default: https://raw.githubusercontent.com/NNielsen1995/Temp/refs/heads/main
        """
        self.data_path = data_path
    
    def extract_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data from CSV files.
        
        Reads 3 datasets:
        - bank_transactions.csv: Customer transactions
        - bank_customers.csv: Customer demographics
        - bank_accounts.csv: Account information
        """        
        transactions = pd.read_csv(f"{self.data_path}/bank_transactions.csv")
        customers = pd.read_csv(f"{self.data_path}/bank_customers.csv")
        accounts = pd.read_csv(f"{self.data_path}/bank_accounts.csv")
        
        return transactions, customers, accounts
    
    def transform_data(
        self, 
        transactions: pd.DataFrame, 
        customers: pd.DataFrame, 
        accounts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform raw data into analytical models
        Steps:
        - Data quality: Remove duplicates and nulls
        - Add derived columns
        - Join datasets into fact table
            
        """
        transactions_clean = transactions.copy()
        transactions_clean = transactions_clean.drop_duplicates(subset=['transaction_id'])
        transactions_clean = transactions_clean.dropna(subset=['amount', 'customer_id'])
                
        #dedup
        customers_clean = customers.copy()
        customers_clean = customers_clean.drop_duplicates(subset=['customer_id'])
        customers_clean = customers_clean.dropna(subset=['customer_id'])
                
        transactions_clean['transaction_date'] = pd.to_datetime(transactions_clean['transaction_date'])
        
        transactions_clean['transaction_month'] = transactions_clean['transaction_date'].dt.to_period('M').astype(str)
        transactions_clean['transaction_year'] = transactions_clean['transaction_date'].dt.year
        transactions_clean['is_high_value'] = (transactions_clean['amount'] > 5000).astype(int)
        
        #Fact table
        fact_transactions = transactions_clean.merge(
            customers_clean, 
            on='customer_id', 
            how='left',
            suffixes=('', '_customer')
        ).merge(
            accounts, 
            on='account_id', 
            how='left',
            suffixes=('', '_account')
        )
        
        fact_transactions = fact_transactions[[
            'transaction_id', 'customer_id', 'account_id',
            'transaction_date', 'transaction_month', 'transaction_year',
            'transaction_type', 'amount', 'merchant_category', 'is_high_value',
            'age', 'city', 'account_type', 'credit_score', 'employment_status'
        ]]
                
        return fact_transactions, customers_clean
    
    def generate_insights(
        self, 
        fact_transactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate insights from transformed data

        Creates 3 outputs:
        - Monthly summary: Transaction volumes by month
        - High value consumers: Top 10% customers by spending
        - Category analysis: Spending by merchant category
        """        
        #Monthly summary
        monthly_summary = fact_transactions.groupby(
            ['transaction_month', 'account_type']
        ).agg(
            transaction_count=('transaction_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_transaction_amount=('amount', 'mean'),
            active_customers=('customer_id', 'nunique')
        ).reset_index()
        
        monthly_summary = monthly_summary.sort_values(['transaction_month', 'account_type'])
                
        #High value customers
        customer_metrics = fact_transactions.groupby(
            ['customer_id', 'age', 'city', 'credit_score', 'employment_status']
        ).agg(
            transaction_count=('transaction_id', 'count'),
            total_spent=('amount', 'sum'),
            high_value_txn_count=('is_high_value', 'sum')
        ).reset_index()
        
        top_10_threshold = customer_metrics['total_spent'].quantile(0.90)
        
        high_value_customers = customer_metrics[
            customer_metrics['total_spent'] >= top_10_threshold
        ].sort_values('total_spent', ascending=False)
        
        high_value_customers['rank'] = high_value_customers['total_spent'].rank(pct=True, ascending=False)
                
        #Merchant category
        category_analysis = fact_transactions.groupby('merchant_category').agg(
            transaction_count=('transaction_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean')
        ).reset_index()
        
        category_analysis = category_analysis.sort_values('total_amount', ascending=False)
                
        return monthly_summary, high_value_customers, category_analysis
    
    def display_insights(
        self, 
        monthly_summary: pd.DataFrame, 
        high_value_customers: pd.DataFrame, 
        category_analysis: pd.DataFrame
    ) -> None:
        """
        Display dataframes
        """       
        print("Monhly transactions")
        print(monthly_summary.head(10).to_string(index=False))
        
        print("Top 10 customers")
        print(high_value_customers.head(10).to_string(index=False))
        
        print("Merchant categories")
        print(category_analysis.head(10).to_string(index=False))
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute pipeline

        """
        try:
            #Extract
            transactions, customers, accounts = self.extract_data()
            
            #Trasnform
            fact_transactions, customers_clean = self.transform_data(
                transactions, customers, accounts
            )
            
            #Generate 
            monthly_summary, high_value_customers, category_analysis = \
                self.generate_insights(fact_transactions)
            
            #Display
            self.display_insights(
                monthly_summary, 
                high_value_customers, 
                category_analysis
            )     
            
            return monthly_summary, high_value_customers, category_analysis
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = BankingData()
    monthly_summary, high_value_customers, category_analysis = pipeline.run()