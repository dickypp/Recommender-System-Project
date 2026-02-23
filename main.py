import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from surprise.model_selection import KFold

file_path = 'user_ratings.csv'
df = pd.read_csv(file_path)

# Rename Columns
df.columns = ['Game_Id', 'Rating', 'User_Id']

np.random.seed(42)
# Unik user
unique_users = df['User_Id'].unique()
# Sampling 10% user
sample_fraction = 0.10
sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * sample_fraction), replace=False)
df_sampled = df[df['User_Id'].isin(sampled_users)]

print(f"Total users: {len(unique_users)}")
print(f"Sampled users: {len(sampled_users)}")
print(f"Sampled interactions: {len(df_sampled)}")

# Preprocessing Data
df_sampled = df_sampled.dropna(subset=['Game_Id', 'Rating', 'User_Id'])
df_sampled = df_sampled.drop_duplicates(subset=['Game_Id', 'User_Id'])
df_sampled['Rating'] = df_sampled['Rating'].round(1)
print(df_sampled.head(10))

# Jumlah Game yang dimainkan setiap user
games_per_user = df_sampled.groupby('User_Id')['Game_Id'].nunique().reset_index(name='Total_Games')
top_users = games_per_user.sort_values(by='Total_Games', ascending=False).head(10)


# Sparsity Data
num_users = df_sampled['User_Id'].nunique()
num_games = df_sampled['Game_Id'].nunique()
num_ratings = len(df_sampled)

sparsity = 1 - (num_ratings / (num_users * num_games))
density = 1 - sparsity
values = [density * 100, sparsity * 100]

# Grafik Sparsity
plt.figure(figsize=(8, 5))
plt.bar(['Density', 'Sparsity'], values, color=['blue', 'red'])
plt.title('Sparsity Level', fontsize=12, pad=15)
plt.ylim(0, 100)

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)

plt.show()

# distribution of ratings
plt.figure(figsize=(7,4))
plt.hist(df_sampled["Rating"], bins=10, edgecolor="black")
plt.title("Distribusi Nilai Rating", fontsize=12)
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Total Ratings per User
plt.figure(figsize=(7,4))
df_sampled["User_Id"].value_counts().plot(kind='hist', bins=50, edgecolor='black')
plt.title("Distribusi Jumlah Rating per User", fontsize=12)
plt.xlabel("Jumlah Rating per User")
plt.ylabel("Jumlah User")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Total Ratings per Game
ratings_per_user = df_sampled.groupby('User_Id').size().reset_index(name='Total')
plt.figure(figsize=(7,4))
df_sampled["Game_Id"].value_counts().plot(kind='hist', bins=50, edgecolor='black')
plt.title("Distribusi Jumlah Rating per Game", fontsize=12)
plt.xlabel("Jumlah Rating per Game")
plt.ylabel("Jumlah Game")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Games Played User
plt.figure(figsize=(7,4))
games_per_user['Total_Games'].plot(kind='hist', bins=50, edgecolor='black')
plt.title("Jumlah Game yang Dimainkan oleh Pengguna", fontsize=12)
plt.xlabel("Jumlah Game")
plt.ylabel("Jumlah Pengguna")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Sumarizing Ratings
rating_summary = df_sampled.groupby('Game_Id')['Rating'].mean()
plt.figure(figsize=(7,4))
plt.hist(rating_summary, bins=10, edgecolor='black')
plt.title('Distribusi Rating', fontsize=12)
plt.xlabel('Rata-Rata Rating')
plt.ylabel('Frekuensi')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# User Statistics
user_stats = ratings_per_user.sort_values(by='Total', ascending=False).head(50)
plt.figure(figsize=(7, 4))
plt.bar(user_stats['User_Id'].astype(str), user_stats['Total'], edgecolor='black', linewidth=0.7)
plt.title('Top 50 Users by Number of Ratings', fontsize=12)
plt.xlabel('User ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=90, fontsize=7)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Model
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(df_sampled[['User_Id', 'Game_Id', 'Rating']], reader)

# Parameter
parameters = {
    'n_factors': [25, 50, 75, 100],
    'n_epochs': [20],
    'lr_all': [0.001, 0.003, 0.005],
    'reg_all': [0.01, 0.03, 0.05],
    'biased': [True]
}

kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Nested Loop
for nf in parameters['n_factors']:
    for ne in parameters['n_epochs']:
        for lr in parameters['lr_all']:
            for reg in parameters['reg_all']:
                for biased in parameters['biased']:
                    params = {
                        'n_factors': nf,
                        'n_epochs': ne,
                        'lr_all': lr,
                        'reg_all': reg,
                        'biased': biased
                    }
                    fold = 1
                    for trainset, testset in kf.split(data):
                        model = SVD(**params)
                        model.fit(trainset)
                        predictions = model.test(testset)
                        rmse = accuracy.rmse(predictions, verbose=False)
                        mae = accuracy.mae(predictions, verbose=False)
                        print(f"Params: {params}, Fold: {fold}, RMSE: {rmse:.4f}")
                        fold += 1

# Top-N Recommendations
def get_top_n(predictions, trainset, user_id, n=10):
    user_inner_id = trainset.to_inner_uid(user_id)

    all_items = set(trainset.all_items())
    rated_items = set(iid for (iid, _) in trainset.ur[user_inner_id])
    unrated_items = all_items - rated_items

    predictions = []
    for iid_inner in unrated_items:
        iid = trainset.to_raw_iid(iid_inner)
        est = model.predict(user_id, iid).est
        predictions.append((iid, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Tampilan Hasil Rekomendasi untuk User Tertentu
user_id = input("Masukkan User ID: ")
trainset = data.build_full_trainset()
top_n_recommendations = get_top_n([], trainset, user_id, n=10)
print(f"Top 10 recommendations for User ID {user_id}:")
for game_id, est_rating in top_n_recommendations:
    print(f"Game ID: {game_id}, Estimated Rating: {est_rating:.2f}")