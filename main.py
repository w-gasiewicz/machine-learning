import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

classes = [2, 5, 10]
features = [2, 5, 10, 20]
chunkSize = [100, 200, 500, 1000]
n_chunks = 400
n_drifts = [2, 5]

weights_2 = [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4]]
weights_5 = [[0.1, 0.1, 0.1, 0.1, 0.6], [0.2, 0.3, 0.1, 0.1, 0.4], [0.1, 0.2, 0.3, 0.2, 0.2]]
weights_10 = [[0.1, 0.15, 0.05, 0.075, 0.075, 0.05, 0.2, 0.1, 0.05, 0.15],
              [0.05, 0.15, 0.1, 0.05, 0.1, 0.15, 0.15, 0.05, 0.1, 0.15],
              [0.075, 0.125, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05, 0.05]]

oscillation_range = [0.3, 0.6, 0.9]

clfs = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
    sl.ensembles.SEA(GaussianNB(), n_estimators=20),
    sl.ensembles.AWE(GaussianNB(), n_estimators=10),
    sl.ensembles.AWE(GaussianNB(), n_estimators=20),
    sl.ensembles.OOB(GaussianNB(), n_estimators=10),
    sl.ensembles.OOB(GaussianNB(), n_estimators=20),
    sl.ensembles.UOB(GaussianNB(), n_estimators=10),
    sl.ensembles.UOB(GaussianNB(), n_estimators=20),
    MLPClassifier(hidden_layer_sizes=(50), max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(50), max_iter=1000)
]

clfs_data = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10)
]

clf_data_names = [
    "SEA_10"
    ]

clf_names = [
    "SEA_10",
    "SEA_20",
    "AWE_10",
    "AWE_20",
    "OOB_10",
    "OOB_20",
    "UOB_10",
    "UOB_20",
    "MLP",
    "MLP"
]

# Nazwy metryk
metrics_names = ["Recall",
                 "BAC",
                 "Precision",
                 "F1 score",
                 "G-mean",
                 "Accuracy"]

# Wybrana metryka
metrics = [sl.metrics.recall,
           sl.metrics.balanced_accuracy_score,
           sl.metrics.precision,
           sl.metrics.f1_score,
           sl.metrics.geometric_mean_score_1,
           accuracy_score]

# for cs in chunkSize:
#     scores = [[]for i in range(len(metrics))]
#     for f in features:#ilosc klas i stopien zbalansowania
#         stream = sl.streams.StreamGenerator(n_chunks=n_chunks,
#                                             chunk_size=cs,
#                                             n_classes=2,
#                                             n_drifts=1,
#                                             n_features=20,
#                                             n_informative=f,
#                                             n_redundant=0,
#                                             n_repeated=0,
#                                             random_state=12345,
#                                             weights=weights_2[1]
#                                             )
#         # Inicjalizacja ewaluatora
#         evaluator = sl.evaluators.TestThenTrain(metrics)
#         # Uruchomienie
#         evaluator.process(stream, clfs_data)
#
#         for m, metric in enumerate(metrics):
#             scores[m].append(evaluator.scores[0, :, m])
#         #print(len(scores[0]))
#
#     # Rysowanie wykresu
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         for i, clf in enumerate(scores[m]):
#             ax[p].plot(scores[m][i], label="data" + str(features[i]) + "_" + str(cs))
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if(p % 2 == 0 and p != 0):
#             plt.savefig("data_" + str(cs)+ "_" + str(m) + ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p+=1

# for cl in classes:#badanie stopnia zbalansowania
#     for nd in n_drifts:
#         clfs_data = [
#             sl.ensembles.SEA(GaussianNB(), n_estimators=10)
#         ]
#         if cl == 2:
#             weightsLoop = weights_2
#         if cl == 5:
#             weightsLoop = weights_5
#         if cl == 10:
#             weightsLoop = weights_10
#         scores = [[]for i in range(len(metrics))]
#         for w in weightsLoop:
#             print(cl)
#             print(w)
#             print(nd)
#             stream = sl.streams.StreamGenerator(n_chunks=n_chunks,
#                                                 chunk_size=500,
#                                                 n_classes=cl,
#                                                 n_drifts=nd,
#                                                 n_features=10,
#                                                 n_informative=10,
#                                                 n_redundant=0,
#                                                 n_repeated=0,
#                                                 random_state=12345,
#                                                 weights=w
#                                                 )
#             # Inicjalizacja ewaluatora
#             evaluator = sl.evaluators.TestThenTrain(metrics)
#             # Uruchomienie
#             evaluator.process(stream, clfs_data)
#
#             for m, metric in enumerate(metrics):
#                 scores[m].append(evaluator.scores[0, :, m])
#
#         # Rysowanie wykresu
#         fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#         p=0
#         for m, metric in enumerate(metrics):
#             ax[p].set_title(metrics_names[m])
#             ax[p].set_ylim(0, 1)
#             for i, clf in enumerate(scores[m]):
#                 ax[p].plot(scores[m][i], label="balance_" + str(nd) + "_" + str(cl)+"_"+str(weightsLoop[i]))
#             plt.ylabel("Metric")
#             plt.xlabel("Chunk")
#             ax[p].legend()
#             if(p % 2 == 0 and p != 0):
#                 plt.savefig("balance_" + str(nd)+ "_" + str(cl) + "_" + str(m) + ".png")
#                 fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#                 p = 0
#             else:
#                 p+=1

# for cl in classes:
#     for nd in n_drifts:
#         clfs_data = [
#             sl.ensembles.SEA(GaussianNB(), n_estimators=10)
#         ]
#         scores = [[]for i in range(len(metrics))]
#         for o in oscillation_range:
#             print(o)
#             print(nd)
#             print(cl)
#             stream = sl.streams.StreamGenerator(n_chunks=n_chunks,
#                                                 chunk_size=500,
#                                                 n_classes=cl,
#                                                 n_drifts=nd,
#                                                 n_features=10,
#                                                 n_informative=10,
#                                                 n_redundant=0,
#                                                 n_repeated=0,
#                                                 random_state=12345,
#                                                 weights=(2, 5, o)
#                                                 )
#             # Inicjalizacja ewaluatora
#             evaluator = sl.evaluators.TestThenTrain(metrics)
#             import time
#             t0 = time.time()
#             # Uruchomienie
#             evaluator.process(stream, clfs_data)
#             print(time.time() - t0)
#             for m, metric in enumerate(metrics):
#                 scores[m].append(evaluator.scores[0, :, m])
#
#         # Rysowanie wykresu
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         print(len(scores[0]))
#         for i, clf in enumerate(scores[m]):
#             ax[p].plot(scores[m][i], label="balance_oscilation_" + str(nd) + "_" + str(cl) + "_" + str(oscillation_range[i]))
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if (p % 2 == 0 and p != 0):
#             plt.savefig("balance_oscilation_" + str(nd) + "_" + str(cl) + "_" + str(m) + ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p += 1

#
# for w in weights_2:
#     stream_prequential = sl.streams.StreamGenerator(n_chunks=n_chunks,
#                                         chunk_size=500,
#                                         n_classes=2,
#                                         n_drifts=2,
#                                         n_features=10,
#                                         n_informative=10,
#                                         n_redundant=0,
#                                         n_repeated=0,
#                                         random_state=12345,
#                                         weights=w
#                                         )
#     stream_ttt = sl.streams.StreamGenerator(n_chunks=n_chunks,
#                                         chunk_size=500,
#                                         n_classes=2,
#                                         n_drifts=2,
#                                         n_features=10,
#                                         n_informative=10,
#                                         n_redundant=0,
#                                         n_repeated=0,
#                                         random_state=12345,
#                                         weights=w
#                                         )
#     # Inicjalizacja ewaluatora
#     evaluator_prequential = sl.evaluators.Prequential(metrics)
#     evaluator_ttt = sl.evaluators.TestThenTrain(metrics)
#     # Uruchomienie
#     evaluator_prequential.process(stream_prequential, clfs_data)
#     evaluator_ttt.process(stream_ttt, clfs_data)
#
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         for i, clf in enumerate(clfs_data):
#             ax[p].plot(evaluator_ttt.scores[i, : , m], label="evaluator_ttt_" + str(w))
#             ax[p].plot(evaluator_prequential.scores[i, : , m], label="evaluator_prequential_" + str(w))
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if (p % 2 == 0 and p != 0):
#             plt.savefig("evaluator_" + str(w)+ str(m) + ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p += 1

#
for w in weights_2:
    stream_prequential = sl.streams.StreamGenerator(n_chunks=n_chunks,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=2,
                                        n_features=10,
                                        n_informative=10,
                                        n_redundant=0,
                                        n_repeated=0,
                                        random_state=12345,
                                        weights=w
                                        )
    stream_ttt = sl.streams.StreamGenerator(n_chunks=n_chunks,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_drifts=2,
                                        n_features=10,
                                        n_informative=10,
                                        n_redundant=0,
                                        n_repeated=0,
                                        random_state=12345,
                                        weights=w
                                        )
    # Inicjalizacja ewaluatora
    evaluator_prequential = sl.evaluators.Prequential(metrics)
    evaluator_ttt = sl.evaluators.TestThenTrain(metrics)
    # Uruchomienie
    evaluator_prequential.process(stream_prequential, clfs)
    print("prequential "+w)
    evaluator_ttt.process(stream_ttt, clfs)
    print("ttt "+ w)
#
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         for i in range(0, len, 2):#enumerate(clfs):
#             ax[p].plot(evaluator_ttt.scores[i, : , m], label="evaluator_ttt_10estimators_" + str(w) + "_" + clf_names[i])
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if (p % 2 == 0 and p != 0):
#             plt.savefig("evaluator_ttt_10_estimators" + str(w)+ "_"+str(m) + "_" + clf_names[i] +  ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p += 1
#
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         for i in range(1, len, 2):  # enumerate(clfs):
#             ax[p].plot(evaluator_ttt.scores[i, :, m], label="evaluator_ttt_20estimators_" + str(w) + "_" + clf_names[i])
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if (p % 2 == 0 and p != 0):
#             plt.savefig("evaluator_ttt_20estimators_" + str(w) +"_"+ str(m) + "_" + clf_names[i] +  ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p += 1
#
#     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#     p = 0
#     for m, metric in enumerate(metrics):
#         ax[p].set_title(metrics_names[m])
#         ax[p].set_ylim(0, 1)
#         for i in range(0, len, 2):
#             ax[p].plot(evaluator_prequential.scores[i, :, m], label="evaluator_prequential_10estimators" + str(w) + "_" + clf_names[i])
#         plt.ylabel("Metric")
#         plt.xlabel("Chunk")
#         ax[p].legend()
#         if (p % 2 == 0 and p != 0):
#             plt.savefig("evaluator_prequential_10estimators_" + str(w) +"_" + str(m) + "_" + clf_names[i] +  ".png")
#             fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#             p = 0
#         else:
#             p += 1
#
#         fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#         p = 0
#         for m, metric in enumerate(metrics):
#             ax[p].set_title(metrics_names[m])
#             ax[p].set_ylim(0, 1)
#             for i in range(1, len, 2):
#                 ax[p].plot(evaluator_prequential.scores[i, :, m], label="evaluator_prequential_20estimators" + str(w) + "_" + clf_names[i])
#             plt.ylabel("Metric")
#             plt.xlabel("Chunk")
#             ax[p].legend()
#             if (p % 2 == 0 and p != 0):
#                 plt.savefig("evaluator_prequential_20estimators_" + str(w) + "_" + str(m) + "_" + clf_names[i] + ".png")
#                 fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#                 p = 0
#             else:
#                 p += 1
#
#     for z in range(0, len, 2):
#         fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#         p = 0
#         for m, metric in enumerate(metrics):
#             ax[p].set_title(metrics_names[m])
#             ax[p].set_ylim(0, 1)
#             for i in range(z, z+2):  # enumerate(clfs):
#                 ax[p].plot(evaluator_ttt.scores[i, :, m], label="evaluator_ttt_estimator_comparison_" + str(w) + "_" + clf_names[i])
#             plt.ylabel("Metric")
#             plt.xlabel("Chunk")
#             ax[p].legend()
#             if (p % 2 == 0 and p != 0):
#                 plt.savefig("evaluator_ttt_estimator_comparison_" + str(w) +"_" + str(m) + "_" + clf_names[i] + ".png")
#                 fig, ax = plt.subplots(1, 3, figsize=(24, 8))
#                 p = 0
#             else:
#                 p += 1