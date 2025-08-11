import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss function"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted loss"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Use weighted cross entropy loss
            weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32, device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Use regular cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class SimplifiedImprovedTrainer:
    """
    Simplified trainer for improved political content detection model
    Fixes serialization issues and focuses on core functionality
    """
    
    def __init__(self, base_dir=None, model_name="distilbert-base-uncased"):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / '.datasets'
        self.results_dir = self.base_dir / 'trainingresults'
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.model_name = model_name
        self.max_length = 256  # Increased for longer political tweets
        self.num_labels = 2
        
        # Training configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def load_improved_datasets(self):
        """Load the improved preprocessed datasets"""
        train_path = self.datasets_dir / 'train_improved.csv'
        val_path = self.datasets_dir / 'val_improved.csv'
        test_path = self.datasets_dir / 'test_improved.csv'
        
        if not all([path.exists() for path in [train_path, val_path, test_path]]):
            raise FileNotFoundError("Improved datasets not found. Run improved_data_preprocessor.py first.")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded improved datasets:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        # Check class distribution
        train_dist = train_df['label'].value_counts()
        logger.info(f"Training set class distribution: {train_dist.to_dict()}")
        
        return train_df, val_df, test_df
    
    def initialize_model_and_tokenizer(self):
        """Initialize tokenizer and model"""
        logger.info(f"Initializing model: {self.model_name}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Move model to device
        self.model.to(self.device)
        
    def calculate_class_weights(self, train_df):
        """Calculate class weights for imbalanced dataset"""
        y_train = train_df['label'].values
        classes = np.unique(y_train)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        logger.info(f"Calculated class weights: {dict(zip(classes, class_weights))}")
        
        return class_weights
    
    def tokenize_datasets(self, train_df, val_df, test_df):
        """Tokenize datasets for training"""
        logger.info("Tokenizing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Will be handled by data collator
                max_length=self.max_length,
            )
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        
        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics during evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        # Also compute per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_non_political': f1_per_class[0],
            'f1_political': f1_per_class[1],
            'precision_non_political': precision_per_class[0],
            'precision_political': precision_per_class[1],
            'recall_non_political': recall_per_class[0],
            'recall_political': recall_per_class[1],
        }
    
    def train_improved_model(self, train_dataset, val_dataset, class_weights=None, output_dir=None):
        """Train the improved political classification model"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"simple_improved_classifier_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simplified training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            
            # Training schedule
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=4,
            
            # Optimizer settings
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            
            # Logging
            logging_dir=str(output_dir / 'logs'),
            logging_steps=200,
            
            # Performance
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),
            
            # Early stopping
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Reproducibility
            seed=42,
            
            # Disable wandb
            report_to=[],
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        
        # Initialize trainer with or without class weights
        if class_weights is not None:
            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[early_stopping],
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[early_stopping],
            )
        
        logger.info("Starting simplified improved model training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving simplified improved model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'training_args': {
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'weight_decay': training_args.weight_decay,
                'warmup_steps': training_args.warmup_steps,
            },
            'class_weights': class_weights.tolist() if class_weights is not None else None,
            'timestamp': datetime.now().isoformat(),
            'dataset_version': 'improved_simplified',
            'final_eval_metrics': trainer.state.log_history[-1] if trainer.state.log_history else {}
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlink to latest improved model
        latest_dir = self.results_dir / 'latest_simple_improved'
        if latest_dir.exists():
            latest_dir.unlink()
        latest_dir.symlink_to(output_dir.name)
        
        return trainer
    
    def evaluate_improved_model(self, trainer, test_dataset, output_dir):
        """Comprehensive evaluation on test set"""
        logger.info("Evaluating simplified improved model on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Non-Political', 'Political'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save evaluation results
        eval_results = {
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'per_class_metrics': {
                'non_political': {
                    'precision': precision_per_class[0],
                    'recall': recall_per_class[0],
                    'f1': f1_per_class[0],
                    'support': int(support_per_class[0])
                },
                'political': {
                    'precision': precision_per_class[1],
                    'recall': recall_per_class[1],
                    'f1': f1_per_class[1],
                    'support': int(support_per_class[1])
                }
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(output_dir / 'test_results_simple_improved.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info("Simplified Improved Model Evaluation Results:")
        logger.info(f"  Test Accuracy: {accuracy:.4f}")
        logger.info(f"  Test F1: {f1:.4f}")
        logger.info(f"  Test Precision: {precision:.4f}")
        logger.info(f"  Test Recall: {recall:.4f}")
        logger.info(f"  Non-Political F1: {f1_per_class[0]:.4f}")
        logger.info(f"  Political F1: {f1_per_class[1]:.4f}")
        
        return eval_results
    
    def full_simple_improved_pipeline(self):
        """Run the complete simplified improved training pipeline"""
        logger.info("Starting simplified improved training pipeline...")
        
        try:
            # Load improved datasets
            train_df, val_df, test_df = self.load_improved_datasets()
            
            # Calculate class weights for imbalanced data
            class_weights = self.calculate_class_weights(train_df)
            
            # Initialize model and tokenizer
            self.initialize_model_and_tokenizer()
            
            # Tokenize datasets
            train_dataset, val_dataset, test_dataset = self.tokenize_datasets(train_df, val_df, test_df)
            
            # Train model with class weights
            trainer = self.train_improved_model(train_dataset, val_dataset, class_weights)
            
            # Get output directory
            output_dir = Path(trainer.args.output_dir)
            
            # Evaluate model
            eval_results = self.evaluate_improved_model(trainer, test_dataset, output_dir)
            
            logger.info("Simplified improved training pipeline completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            
            return trainer, eval_results
            
        except Exception as e:
            logger.error(f"Simplified improved training pipeline failed: {e}")
            raise

def main():
    """Main training function for simplified improved model"""
    logger.info("Training SIMPLIFIED IMPROVED political classifier with distilbert-base-uncased")
    
    trainer_obj = SimplifiedImprovedTrainer(model_name="distilbert-base-uncased")
    trainer, eval_results = trainer_obj.full_simple_improved_pipeline()
    
    print("\n" + "="*60)
    print("SIMPLIFIED IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model: distilbert-base-uncased")
    print(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {eval_results['test_f1']:.4f}")
    print(f"Non-Political F1: {eval_results['per_class_metrics']['non_political']['f1']:.4f}")
    print(f"Political F1: {eval_results['per_class_metrics']['political']['f1']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()