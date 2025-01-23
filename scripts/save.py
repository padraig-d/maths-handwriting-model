from torch import save

def saving(state, filename="trained_model/model.pt"):
    print("Saving checkpoint!")
    save(state, filename)