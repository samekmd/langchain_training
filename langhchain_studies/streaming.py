from model import Model


if __name__ == "__main__":
    model = Model()
    
    resp = model.stream("astronomia")
    for chunk in resp:
        print(chunk)