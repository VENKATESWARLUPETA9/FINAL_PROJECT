from tkinter import messagebox, filedialog, simpledialog
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

main = Tk()
main.title("PREDICTION OF LOAN ELIGIBILITY OF THE CUSTOMER")  # Designing main screen
main.geometry("1300x1200")





# Global Variables
global filename, dataset, X, Y, X_train, X_test, y_train, y_test, le
global classifier, d_classifier, nbclassifier

metrics = {
    "Random Forest": {"precision": 0, "recall": 0, "fscore": 0, "accuracy": 0},
    "Decision Tree": {"precision": 0, "recall": 0, "fscore": 0, "accuracy": 0},
    "Naive Bayes": {"precision": 0, "recall": 0, "fscore": 0, "accuracy": 0},
}

# Functions
def upload():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete("1.0", END)
    text.insert(END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    text.insert(END, str(dataset.head()))
    print(dataset.info())
    sns.set_style("dark")
    dataset.plot(figsize=(18, 8))
    plt.show()


def preprocess():
    global dataset, le
    le = LabelEncoder()
    dataset.drop(["Loan_ID"], axis=1, inplace=True)

    # Encode categorical features
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
        dataset[col] = dataset[col].astype(str)
        dataset[col] = le.fit_transform(dataset[col])

    text.delete("1.0", END)
    text.insert(END, str(dataset.head()))


def splitDataset():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete("1.0", END)

    # Preprocess again in case dataset is updated
    preprocess()

    # Prepare the data for training
    dataset_values = dataset.values
    X = dataset_values[:, :-1]
    Y = dataset_values[:, -1]
    X = normalize(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    text.insert(END, f"Total records found: {X.shape[0]}\n")
    text.insert(END, f"Training records: {X_train.shape[0]}\n")
    text.insert(END, f"Testing records: {X_test.shape[0]}\n")

    # Correlation heatmap
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()


def runModel(algorithm_name, classifier):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Calculate metrics
    metrics[algorithm_name]["accuracy"] = accuracy_score(y_test, predictions) * 100
    metrics[algorithm_name]["precision"] = precision_score(y_test, predictions, average="macro") * 100
    metrics[algorithm_name]["recall"] = recall_score(y_test, predictions, average="macro") * 100
    metrics[algorithm_name]["fscore"] = f1_score(y_test, predictions, average="macro") * 100

    text.delete("1.0", END)
    text.insert(
        END,
        f"{algorithm_name} Accuracy: {metrics[algorithm_name]['accuracy']:.2f}\n"
        f"{algorithm_name} Precision: {metrics[algorithm_name]['precision']:.2f}\n"
        f"{algorithm_name} Recall: {metrics[algorithm_name]['recall']:.2f}\n"
        f"{algorithm_name} F1 Score: {metrics[algorithm_name]['fscore']:.2f}\n\n",
    )


def runRF():
    global classifier
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    runModel("Random Forest", classifier)


def runDecision():
    global d_classifier
    d_classifier = DecisionTreeClassifier()
    runModel("Decision Tree", d_classifier)


def runNBayes():
    global nbclassifier
    nbclassifier = GaussianNB()
    runModel("Naive Bayes", nbclassifier)


def predictEligibility():
    test_file = filedialog.askopenfilename(initialdir="Dataset")
    test_data = pd.read_csv(test_file)
    test_data.fillna(0, inplace=True)
    test_data.drop(["Loan_ID"], axis=1, inplace=True)

    # Encode categorical features
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        test_data[col] = le.transform(test_data[col].astype(str))

    test_data = normalize(test_data.values)
    predictions = classifier.predict(test_data)

    text.delete("1.0", END)
    for i, pred in enumerate(predictions):
        result = "Eligible" if pred == 1 else "Not Eligible"
        text.insert(END, f"Record {i + 1}: {result}\n")


def graph():
    df = pd.DataFrame(
        [
            ["Random Forest", "Precision", metrics["Random Forest"]["precision"]],
            ["Random Forest", "Recall", metrics["Random Forest"]["recall"]],
            ["Random Forest", "F1 Score", metrics["Random Forest"]["fscore"]],
            ["Random Forest", "Accuracy", metrics["Random Forest"]["accuracy"]],
            ["Decision Tree", "Precision", metrics["Decision Tree"]["precision"]],
            ["Decision Tree", "Recall", metrics["Decision Tree"]["recall"]],
            ["Decision Tree", "F1 Score", metrics["Decision Tree"]["fscore"]],
            ["Decision Tree", "Accuracy", metrics["Decision Tree"]["accuracy"]],
            ["Naive Bayes", "Precision", metrics["Naive Bayes"]["precision"]],
            ["Naive Bayes", "Recall", metrics["Naive Bayes"]["recall"]],
            ["Naive Bayes", "F1 Score", metrics["Naive Bayes"]["fscore"]],
            ["Naive Bayes", "Accuracy", metrics["Naive Bayes"]["accuracy"]],
        ],
        columns=["Algorithm", "Metric", "Value"],
    )

    df.pivot(index="Metric", columns="Algorithm", values="Value").plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Performance (%)")
    plt.title("Model Performance Comparison")
    plt.show()


# GUI Components
font = ("times", 16, "bold")
Label(main, text="PREDICTION OF LOAN ELIGIBILITY OF THE CUSTOMER", bg="brown", fg="white", font=font, height=3, width=120).place(x=0, y=5)

font1 = ("times", 13, "bold")
Button(main, text="Upload Loan Dataset", command=upload, font=font1).place(x=50, y=100)
pathlabel = Label(main, bg="brown", fg="white", font=font1)
pathlabel.place(x=360, y=100)

Button(main, text="Preprocess Dataset", command=preprocess, font=font1).place(x=50, y=150)
Button(main, text="Generate Train & Test Data", command=splitDataset, font=font1).place(x=300, y=150)
Button(main, text="Run Random Forest", command=runRF, font=font1).place(x=600, y=150)
Button(main, text="Run Decision Tree", command=runDecision, font=font1).place(x=50, y=200)
Button(main, text="Run Naive Bayes", command=runNBayes, font=font1).place(x=300, y=200)
Button(main, text="Predict Eligibility", command=predictEligibility, font=font1).place(x=550, y=200)
Button(main, text="Performance Graph", command=graph, font=font1).place(x=800, y=200)
Button(main, text="Exit", command=main.destroy, font=font1).place(x=1000, y=200)

text = Text(main, height=18, width=150, font=("times", 12, "bold"))
text.place(x=10, y=250)

main.config(bg="brown")
main.mainloop()
