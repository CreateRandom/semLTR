import pyltr

# Copied from the PyLTR demo script, adjust!

# mock data can be obtained here:

# https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&id=8FEADC23D838BDA8%21107&cid=8FEADC23D838BDA8
# Download MQ2007.rar, select one of the folds and extract the txt data to the data folder

# internally parsed into TX, Ty and Tqids

# TX: 42158 * 46 --> query-doument pair * feature matrix
# Ty: 42158-element vector --> relevance judgments
# Tqids: 42158-element vector --> query ids

with open('../data/train.txt') as trainfile, \
        open('../data/vali.txt') as valifile, \
        open('../data/test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=10,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print ('Our model:', metric.calc_mean(Eqids, Ey, Epred))