import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================
# PROJECT TITLE
# ==============================

PROJECT_TITLE = "Fine Grained Topic Categorization of BBC News Articles using Supervised Document Classification Models"

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("Dataset/bbc_data.csv")

TEXT_COLUMN = "data"
LABEL_COLUMN = "labels"

# Label encoder
encoder = LabelEncoder()
encoder.fit(df[LABEL_COLUMN])

# ==============================
# LOAD MODELS
# ==============================

dt_model = joblib.load("model/decision_tree.pkl")
nb_model = joblib.load("model/naive_bayes.pkl")
knn_model = joblib.load("model/knn_classifier.pkl")
rnc_model = joblib.load("model/radius_neighbors.pkl")

models = {
    "Decision Tree": dt_model,
    "Gaussian NB": nb_model,
    "KNN": knn_model,
    "Radius Neighbor": rnc_model
}

# ==============================
# MODEL METRICS (YOUR RESULTS)
# ==============================

model_metrics = {

"Radius Neighbor":{
"accuracy":0.229,
"precision":0.05,
"recall":0.23,
"f1":0.09
},

"Decision Tree":{
"accuracy":0.8899,
"precision":0.89,
"recall":0.89,
"f1":0.89
},

"Gaussian NB":{
"accuracy":0.93,
"precision":0.93,
"recall":0.93,
"f1":0.93
},

"KNN":{
"accuracy":0.96,
"precision":0.96,
"recall":0.96,
"f1":0.96
}

}

# ==============================
# LOAD GLOVE
# ==============================

print("Loading GloVe embeddings...")

embeddings = {}

with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings[word] = vector

print("GloVe Loaded")

EMBEDDING_DIM = 100


# ==============================
# PREPROCESSING
# ==============================

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z ]','',text)
    tokens = text.split()

    return tokens


# ==============================
# TEXT → VECTOR
# ==============================

def text_vector(text):

    tokens = preprocess_text(text)

    vectors = []

    for word in tokens:
        if word in embeddings:
            vectors.append(embeddings[word])

    if len(vectors) == 0:
        return np.zeros((1,EMBEDDING_DIM))

    vec = np.mean(vectors,axis=0)

    return vec.reshape(1,-1)


# ==============================
# GUI APPLICATION
# ==============================

class BBCApp:

    def __init__(self,root):

        self.root = root
        root.title("BBC AI News Classifier")
        root.geometry("1200x750")

        self.create_layout()


# ==============================
# LAYOUT
# ==============================

    def create_layout(self):

        title = tk.Label(
            self.root,
            text=PROJECT_TITLE,
            font=("Arial",18,"bold"),
            wraplength=1000
        )

        title.pack(pady=10)

        container = tk.Frame(self.root)
        container.pack(fill="both",expand=True)

        sidebar = tk.Frame(container,bg="#2c3e50",width=220)
        sidebar.pack(side="left",fill="y")

        self.main_frame = tk.Frame(container,bg="white")
        self.main_frame.pack(side="right",fill="both",expand=True)

        buttons = [
            ("Dashboard",self.show_dashboard),
            ("Dataset",self.show_dataset),
            ("Preprocessing",self.show_preprocessing),
            ("Train/Test Split",self.show_split),
            ("Algorithms",self.show_algorithms),
            ("Model Comparison",self.show_comparison)
        ]

        for text,cmd in buttons:

            btn = tk.Button(
                sidebar,
                text=text,
                width=22,
                pady=12,
                font=("Arial",10,"bold"),
                command=cmd,
                bg="#34495e",
                fg="white",
                activebackground="#1abc9c"
            )

            btn.pack(pady=5)

        self.show_dashboard()


# ==============================
# CLEAR FRAME
# ==============================

    def clear(self):

        for widget in self.main_frame.winfo_children():
            widget.destroy()


# ==============================
# DASHBOARD
# ==============================

    def show_dashboard(self):

        self.clear()

        frame = tk.Frame(self.main_frame,bg="white",padx=30,pady=20)
        frame.pack()

        tk.Label(
            frame,
            text="BBC News Topic Prediction",
            font=("Arial",18,"bold"),
            bg="white"
        ).pack(pady=10)

        self.textbox = tk.Text(
            frame,
            height=8,
            width=90,
            bd=2,
            relief="solid"
        )

        self.textbox.pack(pady=10)

        btn_frame = tk.Frame(frame,bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame,text="Decision Tree",width=18,
                  command=lambda:self.predict_model("Decision Tree")).grid(row=0,column=0,padx=5)

        tk.Button(btn_frame,text="Gaussian NB",width=18,
                  command=lambda:self.predict_model("Gaussian NB")).grid(row=0,column=1,padx=5)

        tk.Button(btn_frame,text="KNN",width=18,
                  command=lambda:self.predict_model("KNN")).grid(row=0,column=2,padx=5)

        tk.Button(btn_frame,text="Radius Neighbor",width=18,
                  command=lambda:self.predict_model("Radius Neighbor")).grid(row=0,column=3,padx=5)

        tk.Button(btn_frame,text="Predict All Models",
                  width=25,
                  bg="#27ae60",
                  fg="white",
                  command=self.predict_all).grid(row=1,column=1,columnspan=2,pady=10)

        self.result_box = tk.Text(
            frame,
            height=8,
            width=80,
            bd=2,
            relief="solid"
        )

        self.result_box.pack(pady=10)

        tk.Button(
            frame,
            text="Clear Results",
            width=15,
            bg="#e74c3c",
            fg="white",
            command=self.clear_results
        ).pack(pady=5)

        info = f"Dataset Size : {len(df)}     Categories : {list(df[LABEL_COLUMN].unique())}"

        tk.Label(frame,text=info,bg="white").pack(pady=10)


# ==============================
# CLEAR RESULTS
# ==============================

    def clear_results(self):

        self.textbox.delete("1.0",tk.END)
        self.result_box.delete("1.0",tk.END)


# ==============================
# PREDICT MODEL
# ==============================

    def predict_model(self,model_name):

        text = self.textbox.get("1.0",tk.END)

        vec = text_vector(text)

        model = models[model_name]

        pred = model.predict(vec)[0]

        category = encoder.inverse_transform([pred])[0]

        self.result_box.delete("1.0",tk.END)

        self.result_box.insert(tk.END,f"{model_name} Prediction : {category}")


# ==============================
# PREDICT ALL MODELS
# ==============================

    def predict_all(self):

        text = self.textbox.get("1.0",tk.END)

        vec = text_vector(text)

        self.result_box.delete("1.0",tk.END)

        for name,model in models.items():

            pred = model.predict(vec)[0]

            category = encoder.inverse_transform([pred])[0]

            self.result_box.insert(tk.END,f"{name} : {category}\n")


# ==============================
# DATASET VIEW
# ==============================

    def show_dataset(self):

        self.clear()

        tree = ttk.Treeview(self.main_frame)

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col,text=col)
            tree.column(col,width=350)

        for i,row in df.head(200).iterrows():
            tree.insert("",tk.END,values=list(row))

        tree.pack(fill="both",expand=True)


# ==============================
# PREPROCESSING
# ==============================

    def show_preprocessing(self):

        self.clear()

        sample = df[TEXT_COLUMN].iloc[0]

        processed = preprocess_text(sample)

        tk.Label(
            self.main_frame,
            text=f"Original Text:\n\n{sample}\n\nProcessed Tokens:\n\n{processed[:40]}",
            wraplength=900,
            justify="left",
            bg="white"
        ).pack(pady=20)


# ==============================
# TRAIN TEST SPLIT
# ==============================

    def show_split(self):

        self.clear()

        X_train,X_test,y_train,y_test = train_test_split(
            df[TEXT_COLUMN],
            df[LABEL_COLUMN],
            test_size=0.2,
            random_state=42
        )

        info = f"""
Training Samples : {len(X_train)}
Testing Samples : {len(X_test)}
Split Ratio : 80 / 20
"""

        tk.Label(self.main_frame,text=info,font=("Arial",14),bg="white").pack(pady=40)


# ==============================
# ALGORITHMS PAGE
# ==============================

    def show_algorithms(self):

        self.clear()

        frame = tk.Frame(self.main_frame,bg="white")
        frame.pack(pady=30)

        tk.Label(frame,text="Individual Model Performance",
                 font=("Arial",16,"bold"),bg="white").pack(pady=10)

        for model in model_metrics.keys():

            tk.Button(
                frame,
                text=model + " Performance",
                width=30,
                command=lambda m=model:self.show_model_performance(m)
            ).pack(pady=5)


# ==============================
# INDIVIDUAL MODEL GRAPH
# ==============================

    def show_model_performance(self,model_name):

        metrics = model_metrics[model_name]

        labels = ["Accuracy","Precision","Recall","F1 Score"]

        values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"]
        ]

        plt.figure(figsize=(6,4))

        plt.bar(labels,values)

        plt.ylim(0,1)

        plt.title(model_name + " Performance")

        plt.show()


# ==============================
# MODEL COMPARISON
# ==============================

    def show_comparison(self):

        models = list(model_metrics.keys())

        accuracy = [model_metrics[m]["accuracy"] for m in models]
        precision = [model_metrics[m]["precision"] for m in models]
        recall = [model_metrics[m]["recall"] for m in models]
        f1 = [model_metrics[m]["f1"] for m in models]

        x = np.arange(len(models))
        width = 0.2

        plt.figure(figsize=(10,6))

        plt.bar(x-width*1.5,accuracy,width,label="Accuracy")
        plt.bar(x-width/2,precision,width,label="Precision")
        plt.bar(x+width/2,recall,width,label="Recall")
        plt.bar(x+width*1.5,f1,width,label="F1 Score")

        plt.xticks(x,models)

        plt.ylabel("Score")

        plt.title("Model Performance Comparison")

        plt.legend()

        plt.show()


# ==============================
# RUN APP
# ==============================

root = tk.Tk()

app = BBCApp(root)

root.mainloop()