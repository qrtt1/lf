from datasets import load_from_disk


def as_chat(data):
    message = f"""
### HUMAN:
請將引號內的華語改寫，結果的第 1 行為漢字，第 2 行為羅馬字
"{data['華語']}"

### RESPONSE:
"{data['漢字']}"
"{data['羅馬字']}"
    """
    return dict(chat=message.strip())


def main():
    train_dataset = load_from_disk('dataset/train')

    converted = train_dataset.map(as_chat)

    for batch in converted:
        print(batch['chat'])


if __name__ == '__main__':
    main()
