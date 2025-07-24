from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

veri = []
for _ in range(100):
    not1 = random.randint(0, 100)
    not2 = random.randint(0, 100)
    ort = (not1 + not2) / 2
    gecti_mi = 1 if ort >= 50 else 0
    veri.append(([not1, not2], gecti_mi))

X = [v[0] for v in veri]
y = [v[1] for v in veri]

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_egitim, y_egitim)

tahminler = model.predict(X_test)
basari = accuracy_score(y_test, tahminler)
print(f"Model başarı oranı: %{int(basari * 100)}")

print("\nNotlarını gir:")
not1 = int(input("1. not: "))
not2 = int(input("2. not: "))
tahmin = model.predict([[not1, not2]])[0]

if tahmin == 1:
    print("Geçtin")
else:
    print("Kaldın")



