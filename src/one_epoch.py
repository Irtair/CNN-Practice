import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm

class TrainModel():
    def __init__(self, config, model, optimizer, criterion, train_loader, val_loader, device, scheduler):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_vloss = float('inf')
        self.device = device
        self.scheduler = scheduler
        
    def train(self, epoch):
        self.model.train()

        running_loss = 0.

        for data in tqdm(self.train_loader, desc=f"🔃 Тренировка модели | Эпоха {epoch}"):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)


    def val(self, epoch, avg_train_loss):
        # Перевод модели в режим валидации
        self.model.eval()
        running_vloss = 0.0
        all_preds = []
        all_labels = []

        # Валидация
        with torch.no_grad():
            for vinputs, vlabels in tqdm(self.val_loader, desc=f"🔃 Валидация модели | Эпоха {epoch}", leave=False):
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self.model(vinputs)
                vloss = self.criterion(voutputs, vlabels)

                running_vloss += vloss.item()

                preds = torch.argmax(voutputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(vlabels.cpu().numpy())

        avg_vloss = running_vloss / len(self.val_loader)

        class_names = self.val_loader.dataset.classes

        if epoch % 4 == 0:
            # Метрики по каждому классу
            report = classification_report(all_labels, all_preds, target_names=class_names)
            overall_accuracy = accuracy_score(all_labels, all_preds)

            print(f'Train loss: {avg_train_loss:.4f} | Val loss: {avg_vloss:.4f}')
            print(f'Overall Accuracy: {overall_accuracy:.4f}')
            print('\nМетрики по классам:')
            print(report)
        
        self.scheduler.step(avg_vloss)

        # Сохранение лучшей модели
        if avg_vloss < self.best_vloss:
            self.best_vloss = avg_vloss
            torch.save(self.model.state_dict(), f'./best_model/best_model.pt')

        return avg_vloss