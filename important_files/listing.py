edges = {
    [source_service_id]:{
        [target_service_id]: {
            [souce_field_name]: {
                [target_filed_name]: number;
        }
            total_transactions_between_servicies: number;
        }
    }
}

composition = {
    source_iter_number: number,
    source_call_id: number,
    source_service_id: number,
    target_iter_number: number,
    target_call_id: number,
    target_service_id: number,
    fields: {
        source_field_name: string,
        target_field_name: string,
    }[],
} []


composition_tree = {
   [source_service_id]:{
        [target_service_id]: {
            [souce_field_name]: {
                [target_field_name]: composition_tree
            }
        }
    }
}



links = {
    [file_name] : {
        last_serivce_id: number;
        field_name: string;
        current_sequence: composition
    }
}

for (call of logs) {
    for(input_field in call.input) {
        if (!call.is_success) return; 
        for (out_field in call.output) {
            if (links[call.input[input_field]]) {
                add_edge(links[call.input[input_field]].last_serivce_id, call.service_id,input_field, out_field)
            }
            add_link(out_field, call, links[call.input[input_field]] || undefined )
            if (call.type === "publish") {
                add_composition(links[call.output[out_field]])
            }
        }
    }
}

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
print(X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean" )
# Определение количества складок
kfold = KFold(n_splits=10, shuffle=True)

# Выполнение k-fold кросс-валидации
scores = []
for (train_index, test_index) in kfold.split(X, y=y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
knn.predict
# Вычисление средней точности по складкам
mean_accuracy = sum(scores) / len(scores)