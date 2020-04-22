# Helper functions

import matplotlib.pyplot as plt


def sampled(X_train, X_test, y_train, y_test):
    
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import RandomOverSampler
    
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=33)
    X_train, y_train = smote.fit_sample(X_train, y_train) 
    
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_test, y_test = oversample.fit_resample(X_test, y_test)
    
    return X_train, X_test, y_train, y_test

def print_accuracy_report(X_train, X_test, y_train, y_test, model):
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    training_preds = model.predict(X_train)
    y_pred = model.predict(X_test)
    training_accuracy = accuracy_score(y_train, training_preds)
    val_accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
 
    
    print("\n\nTraining Accuracy: {:.4}%".format(training_accuracy * 100))
    print("Validation Accuracy: {:.4}%".format(val_accuracy * 100))
    print("F1 Score: {:.4}%". format(f1))
   
    

    # Classification report
    print("\n\n\nClassification Report:")
    print("---------------------")
    print(classification_report(y_test, y_pred))

    
def plot_con_matrix(y_test, y_pred, class_names, cmap=plt.cm.Blues):
    
    import numpy as np
    import itertools
    from sklearn.metrics import confusion_matrix
    
    plt.grid(b=None)
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.imshow(cnf_matrix, cmap=cmap) 

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add appropriate axis scales
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add labels to each cell
    thresh = cnf_matrix.max() / 2. # Used for text coloring below
    
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment='center',
                 fontsize=16,
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

    # Add a legend
    plt.colorbar()
    plt.show()
    
def plot_roc_curve(fpr, tpr):
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def grid_search(X_train, X_test, y_train, y_test, model, model_name, params, cv=3):
    
    from sklearn.model_selection import GridSearchCV
    # Create grid search object
    gridsearch = GridSearchCV(
        model,
        param_grid = params,
        cv = cv,
        return_train_score=True
    )

    # Fit on data
    gridsearch.fit(X_train, y_train)
    
    best_params = gridsearch.best_params_
    print(f'Optimal parameters: {best_params}')

    print('\n')
    print(f"Training Accuracy: {gridsearch.best_score_ :.2%}")

    print('\n')
    print(f"Test Accuracy: {gridsearch.score(X_test, y_test) :.2%}")
    
    from sklearn.externals import joblib
    model_name = model_name.lower().replace(' ', '_')
    joblib.dump(gridsearch.best_estimator_, f'{model_name}_gridsearch_output_model.pkl', compress = 1)
    print('\n')
    print('Model saved successfully!')

    return best_params

    
    
    

