import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
from torch.nn import DataParallel, MSELoss
from torch.utils.data import DataLoader

from LazyTextDataset import LazyTextDataset

# Run all numpy warnings as errors to catch issues with pearsonr
np.seterr(all='raise')

logger = logging.getLogger(__name__)


class TransformerRegressionTrainer:
    """ Trainer that executes the training, validation, and testing loops
        given all required information such as model, tokenizer, optimizer and so on. """
    def __init__(self,
                 model=None,
                 tokenizer=None,
                 optimizer=None,
                 scheduler=None,
                 train_files=None,
                 valid_files=None,
                 test_files=None,
                 gpu_ids='auto',
                 batch_size=(64, 64, 64),
                 patience=10):
        logging.info(f"Using torch {torch.__version__}")

        self.datasets, self.dataloaders = self._set_data_loaders(train_files,
                                                                 valid_files,
                                                                 test_files,
                                                                 batch_size)
        self.batch_size = batch_size
        self.device, multi_gpu = self._set_device(gpu_ids)

        self.model = model
        # If we can use multi-gpu, set the model to DataParallel
        self.model = DataParallel(model, device_ids=gpu_ids) if multi_gpu else model
        self.tokenizer = tokenizer
        self.criterion = MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_f = None

        self.patience = patience

    @staticmethod
    def _set_data_loaders(train_files, valid_files, test_files, batch_size):
        """ Create datasets and their respective dataloaders.
            See LazyTextDataset.py """
        datasets = {
            'train': LazyTextDataset(train_files) if train_files is not None else None,
            'valid': LazyTextDataset(valid_files) if valid_files is not None else None,
            'test': LazyTextDataset(test_files) if test_files is not None else None
        }

        if train_files:
            logging.info(f"Training set size: {len(datasets['train'])}")
        if valid_files:
            logging.info(f"Validation set size: {len(datasets['valid'])}")
        if test_files:
            logging.info(f"Test set size: {len(datasets['test'])}")

        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size[0], shuffle=False)
            if train_files is not None else None,
            'valid': DataLoader(datasets['valid'], batch_size=batch_size[1], shuffle=False)
            if valid_files is not None else None,
            'test': DataLoader(datasets['test'], batch_size=batch_size[2], shuffle=False)
            if test_files is not None else None
        }

        return datasets, dataloaders

    @staticmethod
    def _set_device(gpu_ids):
        """ Set current device to use. If gpu_ids is None (null in JSON) the CPU will be used.
            If not None, a list should be given of GPU IDS, e.g. [0] or [0, 1].

            :returns the current main device; whether or not to use multi-gpu
        """
        if gpu_ids is None:
            main_device = torch.device('cpu')
        else:
            main_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        multi_gpu = False
        if main_device.type == 'cuda':
            if (1 < len(gpu_ids) <= torch.cuda.device_count() and isinstance(gpu_ids, list)) \
                    or (gpu_ids == 'auto' and 1 < torch.cuda.device_count()):
                logger.info(f"Using {len(gpu_ids)} GPUs")
                multi_gpu = True
            else:
                device_id = torch.cuda.current_device()
                logger.info(f"Using GPU {torch.cuda.get_device_name(device_id)}")
        else:
            logger.info('Using CPU')

        return main_device, multi_gpu

    @staticmethod
    def prepare_lines(data, cast_to=None):
        """ Basic line preparation, strips away new lines. Can also cast input to datatype, e.g.
            `cast_to=float` when you want your labels as floating point numbers.
            """
        if cast_to:
            out = [cast_to(line.strip()) for line in data]
        else:
            out = [line.strip() for line in data]

        return out

    def save_model(self, valid_loss, valid_pearson, epoch):
        """ Saves current model as well as additional information. """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_pearson': valid_pearson,
            'epoch': epoch
        }, self.checkpoint_f)

    def load_model(self, checkpoint_f):
        """ Load checkpoint, especially used for testing. """
        chckpnt_f = checkpoint_f if checkpoint_f is not None else self.checkpoint_f
        checkpoint = torch.load(chckpnt_f, map_location=self.device)

        # Not all checkpoints might be structured in the same way as we save it
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            self.model.load_state_dict(checkpoint)

    @staticmethod
    def _plot_training(train_losses, valid_losses):
        """ Plot loss into plt graph.
            :returns the figure object of the graph """
        fig = plt.figure(dpi=300)
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.xlabel('epochs')
        plt.legend(frameon=False)
        plt.title('Loss progress')
        # Set ticks to integers rather than floats
        xint = range(len(train_losses))
        plt.xticks(xint)
        plt.show()

        return fig

    def train(self, epochs=10, checkpoint_f='checkpoint.pth', log_update_freq=10):
        """ Entry point to start training the model. Will run the outer epoch loop containing
            training and validation. Also implements early stopping, set by `self.patience`.
            Actual training/validating is done in `self._process()`

        log_update_freq: show a log message every X percent in a batch's progress.
            E.g. for a value of 25, 4 messages will be printed per batch (100/25=4)
        """
        logging.info('Training started.')
        train_start = time.time()

        self.checkpoint_f = checkpoint_f

        valid_loss_min = np.inf
        train_losses, valid_losses = [], []
        last_saved_epoch = 0

        total_train_time = 0
        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_results = self._process('train', log_update_freq, epoch)
            total_train_time += time.time() - epoch_start

            # In the rare case where all results are identical, Pearson calculation will fail
            try:
                train_pearson = pearsonr(train_results['predictions'], train_results['labels'])
            except FloatingPointError:
                train_pearson = "Could not calculate Pearsonr"

            # Calculate average losses
            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # VALIDATION
            valid_loss, valid_results = self._process('valid', log_update_freq, epoch)

            try:
                valid_pearson = pearsonr(valid_results['predictions'], valid_results['labels'])
            except FloatingPointError:
                valid_pearson = "Could not calculate Pearsonr"

            valid_loss = np.mean(valid_loss)
            valid_losses.append(valid_loss)

            # Log epoch statistics
            logging.info(f"Epoch {epoch} - completed in {(time.time() - epoch_start):.0f} seconds"
                         f"\nTraining Loss: {train_loss:.6f}\t Pearson: {train_pearson}"
                         f"\nValidation loss: {valid_loss:.6f}\t Pearson: {valid_pearson}")

            # Save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                logging.info(f'!! Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).')
                logging.info(f'!! Saving model as {self.checkpoint_f}...')

                self.save_model(valid_loss, valid_pearson, epoch)
                last_saved_epoch = epoch
                valid_loss_min = valid_loss
            else:
                logging.info(
                    f"!! Valid loss not improved. (Min. = {valid_loss_min}; last save at ep. {last_saved_epoch})")
                if train_loss <= valid_loss:
                    logging.warning(f"!! Training loss is lte validation loss. Might be overfitting!")

            # Early-stopping
            if self.patience:
                if (epoch - last_saved_epoch) == self.patience:
                    logging.info(f"Stopping early at epoch {epoch} (patience={self.patience})...")
                    break

            # Optimise with scheduler
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

        fig = self._plot_training(train_losses, valid_losses)

        logging.info(f"Training completed in {(time.time() - train_start):.0f} seconds"
                     f"\nMin. valid loss: {valid_loss_min}\nLast saved epoch: {last_saved_epoch}"
                     f"\nPerformance: {len(self.datasets['train']) // total_train_time:.0f} sentences/s")

        return self.checkpoint_f, fig

    def _process(self, do, log_update_freq, epoch=None):
        """ Runs the training, validation, or testing (for one epoch) """
        if do not in ('train', 'valid', 'test'):
            raise ValueError("Use 'train', 'valid', or 'test' for 'do'.")

        results = {'predictions': np.array([]), 'labels': np.array([])}
        losses = np.array([])

        self.model = self.model.to(self.device)
        if do == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if log_update_freq:
            nro_batches = len(self.datasets[do]) // self.dataloaders[do].batch_size
            update_interval = nro_batches * (log_update_freq / 100)
            update_checkpoints = {int(nro_batches - (i * update_interval)) for i in range((100 // log_update_freq))}

        # Main loop: iterate over dataloader
        for batch_idx, data in enumerate(self.dataloaders[do], 1):
            # 0. Clear gradients
            if do == 'train':
                self.optimizer.zero_grad()

            # 1. Data prep
            sentences = data['sentences']
            sentences = self.prepare_lines(sentences)
            input_ids, input_mask = self.tokenizer(sentences)
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            # Convert labels to float array
            labels = data['labels']
            labels = self.prepare_lines(labels, cast_to=float)
            labels = torch.FloatTensor(labels).to(self.device)

            # 2. Predictions
            preds = self.model(input_ids, labels, attention_mask=input_mask)
            loss = self.criterion(preds, labels)

            # 3. Optimise during training
            if do == 'train':
                loss.backward()
                self.optimizer.step()

            # 4. Save results
            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            results['predictions'] = np.append(results['predictions'], preds, axis=None)
            results['labels'] = np.append(results['labels'], labels, axis=None)
            losses = np.append(losses, float(loss))

            # Log progress
            if log_update_freq and batch_idx in update_checkpoints:
                if do in ('train', 'valid'):
                    logging.info(f"{do.capitalize()} epoch {epoch}, batch nr. {batch_idx}/{nro_batches}...")
                else:
                    logging.info(f"{do.capitalize()}, batch nr. {batch_idx}/{nro_batches}...")

        return losses, results

    def test(self, checkpoint_f=None, log_update_freq=0):
        """ Wraps testing a given model. Actual testing is done in `self._process()`. """
        logging.info('Testing started.')
        test_start = time.time()

        self.load_model(checkpoint_f)
        self.model = self.model.to(self.device)

        test_loss, test_results = self._process('test', log_update_freq)

        try:
            test_pearson = pearsonr(test_results['predictions'], test_results['labels'])
        except FloatingPointError:
            test_pearson = "Could not calculate Pearsonr"

        test_loss = np.mean(test_loss)

        logging.info(f"Testing completed in {(time.time() - test_start):.0f} seconds"
                     f"\nLoss: {test_loss:.6f}\t Pearson: {test_pearson}\n")

        return test_loss, test_pearson[0]
