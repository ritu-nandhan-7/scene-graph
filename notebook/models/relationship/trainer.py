from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class RelationshipTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = torch.device(device)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=learning_rate,
            

        )
        self.best_accuracy = 0.0

        self.best_loss = float("inf")
    
    
    def train_one_epoch(
        self,
    ):

        self.model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for batch in self.train_loader:

            # ------------------------------------------
            # Move batch to device
            # ------------------------------------------

            image = batch["image"].to(
                self.device
            )

            subject_label = batch["subject_label"].to(
                self.device
            )

            object_label = batch["object_label"].to(
                self.device
            )

            subject_bbox = batch["subject_bbox"].to(
                self.device
            )

            object_bbox = batch["object_bbox"].to(
                self.device
            )

            target = batch["target"].to(
                self.device
            )

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            logits = self.model(

                image,

                subject_label,

                object_label,

                subject_bbox,

                object_bbox,

            )

            loss = self.criterion(

                logits,

                target,

            )

            # ------------------------------------------
            # Backpropagation
            # ------------------------------------------

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            total_loss += loss.item()

            _, predictions = torch.max(
                logits,
                dim=1,
            )

            correct += (
                predictions == target
            ).sum().item()

            total += target.size(0)

        average_loss = total_loss / len(
            self.train_loader
        )

        accuracy = correct / total

        return average_loss, accuracy
    
    def validate(
        self,
    ):

        self.model.eval()

        total_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for batch in self.val_loader:

                # ------------------------------------------
                # Move batch to device
                # ------------------------------------------

                image = batch["image"].to(
                    self.device
                )

                subject_label = batch["subject_label"].to(
                    self.device
                )

                object_label = batch["object_label"].to(
                    self.device
                )

                subject_bbox = batch["subject_bbox"].to(
                    self.device
                )

                object_bbox = batch["object_bbox"].to(
                    self.device
                )

                target = batch["target"].to(
                    self.device
                )

                # ------------------------------------------
                # Forward
                # ------------------------------------------

                logits = self.model(

                    image,

                    subject_label,

                    object_label,

                    subject_bbox,

                    object_bbox,

                )

                loss = self.criterion(

                    logits,

                    target,

                )

                # ------------------------------------------
                # Statistics
                # ------------------------------------------

                total_loss += loss.item()

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                correct += (
                    predictions == target
                ).sum().item()

                total += target.size(0)

        average_loss = total_loss / len(
            self.val_loader
        )

        accuracy = correct / total

        return average_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        checkpoint_path: str = "best_model.pth",
    ):

        for epoch in range(num_epochs):

            print("=" * 70)

            print(
                f"Epoch {epoch + 1}/{num_epochs}"
            )

            print("=" * 70)

            # ------------------------------------------
            # Training
            # ------------------------------------------

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )

            # ------------------------------------------
            # Validation
            # ------------------------------------------

            val_loss, val_accuracy = (
                self.validate()
            )

            # ------------------------------------------
            # Print Statistics
            # ------------------------------------------

            print(
                f"Train Loss     : {train_loss:.4f}"
            )

            print(
                f"Train Accuracy : {train_accuracy:.4f}"
            )

            print(
                f"Val Loss       : {val_loss:.4f}"
            )

            print(
                f"Val Accuracy   : {val_accuracy:.4f}"
            )

            # ------------------------------------------
            # Save Best Model
            # ------------------------------------------

            if val_accuracy > self.best_accuracy:

                self.best_accuracy = val_accuracy

                self.best_loss = val_loss

                self.save_checkpoint(
                    checkpoint_path
                )

                print(
                    "Best model updated."
                )

            print()
            
    def save_checkpoint(
        self,
        path: str,
    ):

        checkpoint = {

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "best_accuracy": self.best_accuracy,

            "best_loss": self.best_loss,

        }

        torch.save(
            checkpoint,
            path,
        )
    
    def load_checkpoint(
        self,
        path: str,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.best_accuracy = checkpoint[
            "best_accuracy"
        ]

        self.best_loss = checkpoint[
            "best_loss"
        ]
        
    