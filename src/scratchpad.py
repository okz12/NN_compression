### Functions that were ultimately not needed. Saved here in case needed in the future.

### Freezing layers
def freeze_layer(model, layer_n, freeze = True):
    freeze_params = [(layer_n - 1) * 2, (layer_n - 1) * 2 + 1]
    for i,param in enumerate(model.parameters()):
        if i in freeze_params:
            print (param.size(), "frozen" if freeze else "unfrozen")
            param.requires_grad = not freeze


for i,param in enumerate(model.parameters()):
    print param.requires_grad
    
### Trimming Layers
#6. Trimming layers
model = torch.load(model_dir + 'mnist_{}_{}.m'.format(model.name, num_epochs))
result_list = []
for prune_l1 in range(0,101,10):
    L1 = layer_utils(model.fc1.weight.data)
    model.fc1.weight.data = L1.prune(prune_l1)
    L2 = layer_utils(model.fc2.weight.data)
    model.fc2.weight.data = L2.prune(prune_l1)
    L3 = layer_utils(model.fc3.weight.data)
    model.fc3.weight.data = L3.prune(prune_l1)
    accuracy = test_accuracy(test_loader, model)
    print accuracy
    
    
#Exponential decay schedule
def get_decay(decay_start, decay_end, retraining_epochs):
    decay_range = decay_end/decay_start
    decay_step = np.log(decay_range)/retraining_epochs
    decay_schedule = decay_end + decay_start * (1 - np.exp( np.arange(retraining_epochs, 0, -1) * decay_step ) )
    #decay_schedule = min(decay_schedule) - decay_schedule
    plt.plot(decay_schedule)
    plt.show()
    return decay_schedule