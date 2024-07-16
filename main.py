import os
import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

from google.colab import drive

# Google Driveをマウント
drive.mount('/content/drive')

# データセットのパスを設定
DATA_DIR = '/content/drive/MyDrive/VQA_dataset'  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, is_train=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.is_train = is_train
        
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split()
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}
        
        if self.is_train and "answers" in self.df.columns:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df['image'][idx]))
        image = self.transform(image)
        
        question = np.zeros(len(self.idx2question) + 1)
        question_words = process_text(self.df["question"][idx]).split()
        for word in question_words:
            if word in self.question2idx:
                question[self.question2idx[word]] = 1
            else:
                question[-1] = 1
        
        if self.is_train and "answers" in self.df.columns:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class VQAModel(nn.Module):
    def __init__(self, vocab_size, n_answer):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(2048, 512)
        self.attention = Attention(512)
        self.question_encoder = nn.Linear(vocab_size, 512)
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_answer)
        )

    def forward(self, image, question):
        batch_size = image.size(0)
        image_features = self.resnet(image)
        image_features = image_features.view(batch_size, 512, -1).permute(0, 2, 1)
        question_features = self.question_encoder(question)
        
        attn_weights = self.attention(question_features, image_features)
        context = torch.bmm(attn_weights.unsqueeze(1), image_features).squeeze(1)
        x = torch.cat([context, question_features], dim=1)
        x = self.fc(x)
        return x

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for i, batch in enumerate(dataloader):
        try:
            image, question, answers, mode_answer = batch
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
            
            with autocast():
                pred = model(image, question)
                loss = criterion(pred, mode_answer)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)
            simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
            
            if i % 10 == 0:
                print(f"Batch {i}/{len(dataloader)}: Loss: {loss.item():.4f}")
        except RuntimeError as e:
            print(f"Error during training: {e}")
            print(f"Image shape: {image.shape}, Question shape: {question.shape}")
            continue
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            try:
                image, question, answers, mode_answer = batch
                image, question, answer, mode_answer = \
                    image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
                
                pred = model(image, question)
                loss = criterion(pred, mode_answer)
                
                total_loss += loss.item()
                total_acc += VQA_criterion(pred.argmax(1), answers)
                simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
                
                if i % 10 == 0:
                    print(f"Validation Batch {i}/{len(dataloader)}: Loss: {loss.item():.4f}")
            except RuntimeError as e:
                print(f"Error during validation: {e}")
                print(f"Image shape: {image.shape}, Question shape: {question.shape}")
                continue
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    train_dataset = VQADataset(df_path=os.path.join(DATA_DIR, "train.json"), image_dir=os.path.join(DATA_DIR, "train"), transform=transform, is_train=True)
    val_dataset = VQADataset(df_path=os.path.join(DATA_DIR, "valid.json"), image_dir=os.path.join(DATA_DIR, "valid"), transform=transform, is_train=False)
    val_dataset.update_dict(train_dataset)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # 小規模テスト用のオプション
    test_run = False  # Trueに設定すると小規模テストを実行
    if test_run:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(100))
        print("Running in test mode with reduced dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    print("Initializing model...")
    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    num_epoch = 30 if not test_run else 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()

    print("Starting training...")
    for epoch in range(num_epoch):
        try:
            print(f"Epoch {epoch + 1}/{num_epoch}")
            train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, scaler)
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Simple Acc: {train_simple_acc:.4f}, Time: {train_time:.2f}s")
            
            val_loss, val_acc, val_simple_acc = validate(model, val_loader, criterion, device)
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Simple Acc: {val_simple_acc:.4f}")
            
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(DATA_DIR, f"model_epoch_{epoch+1}.pth"))
                print(f"Saved model at epoch {epoch + 1}")
        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {e}")
            continue

    print("Training completed. Starting evaluation...")
    model.eval()
    submission = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                image, question = batch
                image, question = image.to(device), question.to(device)
                pred = model(image, question)
                pred = pred.argmax(1).cpu().numpy()
                submission.extend(pred)
                if i % 10 == 0:
                    print(f"Processed {i}/{len(val_loader)} validation batches")
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save(os.path.join(DATA_DIR, "submission.npy"), submission)
    print("Submission file saved.")

if __name__ == "__main__":
    main()
