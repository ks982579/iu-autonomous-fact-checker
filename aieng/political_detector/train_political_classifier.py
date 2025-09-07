"""
Experiment - but returned back to the notebook approach
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

IS_PLOTTING = True

if IS_PLOTTING:
    import matplotlib.pyplot as plt
    import seaborn as sns

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

class PoliticalClassifierTrainer:
    """
    Trainer for political content detection model
    Optimized for small, efficient models with high performance
    """
    
    def __init__(self, base_dir=None, model_name="distilbert-base-uncased"):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / '.datasets'
        self.results_dir = self.base_dir / 'trainingresults'
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.model_name = model_name  # DistilBERT for efficiency
        self.max_length = 128  # Shorter length for efficiency
        self.num_labels = 2
        
        # Training configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def load_datasets(self):
        """
        Load preprocessed datasets
        
        Returns:
            Tuple: train, val, test DataFrames
        """
        train_path = self.datasets_dir / 'train.csv'
        val_path = self.datasets_dir / 'val.csv'
        test_path = self.datasets_dir / 'test.csv'
        
        if not all([path.exists() for path in [train_path, val_path, test_path]]):
            raise FileNotFoundError("Preprocessed datasets not found. Run data_preprocessor.py first.")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded datasets:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def initialize_model_and_tokenizer(self):
        """
        Initialize tokenizer and model
        """
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
        
    def tokenize_datasets(self, train_df, val_df, test_df):
        """
        Tokenize datasets for training
        
        Args:
            train_df, val_df, test_df: DataFrames with 'text' and 'label' columns
            
        Returns:
            Tuple: tokenized train, val, test datasets
        """
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
        """
        Compute metrics during evaluation
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir=None):
        """
        Train the political classification model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for model
            
        Returns:
            Trainer: Trained model trainer
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"political_classifier_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments optimized for efficiency
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            
            # Training schedule
            num_train_epochs=3,  # Start with fewer epochs
            per_device_train_batch_size=32,  # Larger batch size for efficiency
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=2,  # Effective batch size = 64
            
            # Optimizer settings
            learning_rate=2e-5,  # Standard learning rate for fine-tuning
            weight_decay=0.01,
            warmup_steps=500,
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,  # Keep only best 3 checkpoints
            load_best_model_at_end=True,
            
            # Logging
            logging_dir=str(output_dir / 'logs'),
            logging_steps=100,
            
            # Performance
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),  # Mixed precision for speed
            
            # Early stopping
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Reproducibility
            seed=42,
            
            # Disable wandb if not needed
            report_to=[],
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        
        # Initialize trainer
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
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'training_args': training_args.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'final_eval_metrics': trainer.state.log_history[-1] if trainer.state.log_history else {}
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlink to latest
        latest_dir = self.results_dir / 'latest'
        if latest_dir.exists():
            latest_dir.unlink()
        latest_dir.symlink_to(output_dir.name)
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset, output_dir):
        """
        Comprehensive evaluation on test set
        
        Args:
            trainer: Trained model trainer
            test_dataset: Test dataset
            output_dir: Directory to save evaluation results
        """
        logger.info("Evaluating on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
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
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Plot confusion matrix if plotting is available
        if IS_PLOTTING:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Non-Political', 'Political'],
                        yticklabels=['Non-Political', 'Political'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        else:
            logger.info("Plotting libraries not available, skipping confusion matrix plot")
        
        logger.info("Evaluation Results:")
        logger.info(f"  Test Accuracy: {accuracy:.4f}")
        logger.info(f"  Test F1: {f1:.4f}")
        logger.info(f"  Test Precision: {precision:.4f}")
        logger.info(f"  Test Recall: {recall:.4f}")
        
        return eval_results
    
    def analyze_model_size(self, output_dir):
        """
        Analyze and log model size information
        
        Args:
            output_dir: Model output directory
        """
        model_path = Path(output_dir)
        
        # Calculate model file sizes
        size_info = {}
        for file_path in model_path.glob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                size_info[file_path.name] = f"{size_mb:.2f} MB"
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': sum(float(size.split()[0]) for size in size_info.values() if 'MB' in size),
            'file_sizes': size_info
        }
        
        with open(model_path / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Model Size Analysis:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: {model_info['model_size_mb']:.2f} MB")
        
        return model_info
    
    def full_training_pipeline(self):
        """
        Run the complete training pipeline
        """
        logger.info("Starting full training pipeline...")
        
        try:
            # Load datasets
            train_df, val_df, test_df = self.load_datasets()
            
            # Initialize model and tokenizer
            self.initialize_model_and_tokenizer()
            
            # Tokenize datasets
            train_dataset, val_dataset, test_dataset = self.tokenize_datasets(train_df, val_df, test_df)
            
            # Train model
            trainer = self.train_model(train_dataset, val_dataset)
            
            # Get output directory
            output_dir = Path(trainer.args.output_dir)
            
            # Evaluate model
            eval_results = self.evaluate_model(trainer, test_dataset, output_dir)
            
            # Analyze model size
            model_info = self.analyze_model_size(output_dir)
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            
            return trainer, eval_results, model_info
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """
    Main training function
    """
    # Choose model - DistilBERT for efficiency, or bert-base-uncased for higher accuracy
    model_options = {
        'distilbert': 'distilbert-base-uncased', # Fast, smaller
        'bert': 'bert-base-uncased', # More accurate, larger
        'roberta': 'roberta-base', # High performance
    }
    
    # Select model (you can change this)
    selected_model = model_options['distilbert']  # Change to 'bert' or 'roberta' if needed
    
    logger.info(f"Training political classifier with {selected_model}")
    
    trainer_obj = PoliticalClassifierTrainer(model_name=selected_model)
    trainer, eval_results, model_info = trainer_obj.full_training_pipeline()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Model: {selected_model}")
    print(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {eval_results['test_f1']:.4f}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print("="*50)

if __name__ == "__main__":
    main()