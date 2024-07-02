import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import accuracy
from collections import defaultdict


file_path = 'C:/Users/Haroon Virk/Downloads/Data Analysis/TSK-000-192/shopping_trends_updated.csv'

data = pd.read_csv(file_path)


label_encoders = {}
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 
                       'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 
                       'Payment Method', 'Frequency of Purchases']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


scaler = StandardScaler()
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


def reverse_label_encoding(data, label_encoders, columns):
    for col in columns:
        le = label_encoders[col]
        data[col] = le.inverse_transform(data[col])
    return data

data = reverse_label_encoding(data, label_encoders, categorical_columns)


sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.histplot(data['Age'], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Customer Age')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

sns.countplot(x='Gender', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Customer Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')

sns.countplot(x='Frequency of Purchases', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Frequency of Purchases')
axes[1, 0].set_xlabel('Frequency of Purchases')
axes[1, 0].set_ylabel('Count')

sns.countplot(x='Payment Method', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Payment Methods')
axes[1, 1].set_xlabel('Payment Method')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.countplot(x='Subscription Status', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Subscription Status')
axes[0, 0].set_xlabel('Subscription Status')
axes[0, 0].set_ylabel('Count')

sns.countplot(y='Item Purchased', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Most Popular Items')
axes[0, 1].set_xlabel('Count')
axes[0, 1].set_ylabel('Item Purchased')

sns.countplot(x='Category', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Most Popular Categories')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Count')

sns.countplot(x='Season', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Purchases by Season')
axes[1, 1].set_xlabel('Season')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x='Season', y='Purchase Amount (USD)', data=data, ax=axes[0])
axes[0].set_title('Purchase Amount by Season')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Purchase Amount (USD)')

# Compute correlation matrix for numerical columns only
numerical_data = data[numerical_columns]
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1])
axes[1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.scatterplot(x='Age', y='Purchase Amount (USD)', hue='Gender', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Customer Segmentation by Age and Purchase Amount')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Purchase Amount (USD)')

location_purchases = data['Location'].value_counts()
sns.barplot(x=location_purchases.index, y=location_purchases.values, ax=axes[0, 1])
axes[0, 1].set_title('Purchases by Location')
axes[0, 1].set_xlabel('Location')
axes[0, 1].set_ylabel('Number of Purchases')
axes[0, 1].tick_params(axis='x', rotation=90)

sns.histplot(data['Review Rating'], bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Review Ratings')
axes[1, 0].set_xlabel('Review Rating')
axes[1, 0].set_ylabel('Frequency')

sns.boxplot(x='Category', y='Review Rating', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Review Ratings by Category')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Review Rating')

plt.tight_layout()
plt.show()

avg_purchase_by_category = data.groupby('Category')['Purchase Amount (USD)'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_purchase_by_category.index, y=avg_purchase_by_category.values)
plt.title('Average Purchase Amount by Category')
plt.xlabel('Category')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=90)
plt.show()

avg_rating_by_item = data.groupby('Item Purchased')['Review Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_rating_by_item.index, y=avg_rating_by_item.values)
plt.title('Average Review Rating by Item Purchased')
plt.xlabel('Item Purchased')
plt.ylabel('Average Review Rating')
plt.xticks(rotation=90)
plt.show()

# Recommendation System Evaluation
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(data[['User ID', 'Item Purchased', 'Review Rating']], reader)
trainset, testset = train_test_split(data_surprise, test_size=0.2)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

# Calculate and print RMSE
rmse = accuracy.rmse(predictions)
# Calculate and print MAE
mae = accuracy.mae(predictions)

# Precision and Recall at k
def precision_recall_at_k(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
    return avg_precision, avg_recall

precision, recall = precision_recall_at_k(predictions, k=5, threshold=3.5)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
