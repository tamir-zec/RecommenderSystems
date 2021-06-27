import datetime as dt
import time

INF = 10000.0


def train(model, criterion, optimizer, reader, hyper_params):
    import torch

    model.train()
    # Initializing metrics since we will calculate RMSE on the train set on the fly
    metrics = {}
    metrics['RMSE'] = 0.0
    # Initializations
    total_x, total_batches = 0.0, 0.0

    # Train for one epoch, batch-by-batch
    for data, y in reader.iter():
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
        # Forward pass
        all_output = model(data)
        # Backward pass
        # loss = criterion(all_output, y, return_mean=False)
        loss = criterion(all_output, y)
        metrics['RMSE'] += float(torch.sum(loss.data))
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        try:
            total_x += float(int(all_output.shape[0]))  # For every model
        except:
            total_x += float(int(all_output[0].shape[0]))  # For TransNet
        total_batches += 1

    metrics['RMSE'] = round(metrics['RMSE'] / float(total_x), 4)

    return metrics


def train_complete(hyper_params, Model, train_reader, val_reader, user_count, item_count, model, review=True):
    import torch

    from loss import RMSELoss
    from eval import evaluate
    from utils import file_write, is_cuda_available, log_end_epoch

    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(len(val_reader)))

    # criterion = MSELoss(hyper_params)
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])

    file_write(hyper_params['log_file'], str(model))
    file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

    try:
        best_RMSE = float(INF)
        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            # Training for one epoch
            metrics = train(model, criterion, optimizer, train_reader, hyper_params)
            metrics['dataset'] = hyper_params['dataset']
            # log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on = '(TRAIN)')

            # Calculating the metrics on the validation set
            metrics, _, _ = evaluate(
                model, criterion, val_reader, hyper_params,
                user_count, item_count, review=review
            )
            metrics['dataset'] = hyper_params['dataset']
            log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on='(VAL)')

            # Save best model on validation set
            if metrics['RMSE'] < best_RMSE:
                print("Saving model...")
                torch.save(model.state_dict(), hyper_params['model_path'])
                best_RMSE = metrics['RMSE']

    except KeyboardInterrupt:
        print('Exiting from training early')

    # Load best model and return it for evaluation on test-set
    model = Model(hyper_params)
    if is_cuda_available: model = model.cuda()
    model.load_state_dict(torch.load(hyper_params['model_path']))
    model.eval()

    return model


def main_pytorch(hyper_params, gpu_id=None):
    from data import load_data
    from eval import evaluate, eval_ranking
    from utils import is_cuda_available
    from utils import load_user_item_counts, xavier_init, log_end_epoch
    from loss import RMSELoss

    if hyper_params['model_type'] in ['deepconn', 'deepconn++']:
        from pytorch_models.DeepCoNN import DeepCoNN as Model
    elif hyper_params['model_type'] in ['transnet', 'transnet++']:
        from pytorch_models.TransNet import TransNet as Model
    elif hyper_params['model_type'] in ['NARRE']:
        from pytorch_models.NARRE import NARRE as Model
    elif hyper_params['model_type'] in ['bias_only', 'MF', 'MF_dot']:
        from pytorch_models.MF import MF as Model

    # Load the data readers
    user_count, item_count = load_user_item_counts(hyper_params)
    if hyper_params['model_type'] not in ['bias_only', 'MF', 'MF_dot', 'NeuMF']:
        review_based_model = True
        try:
            from data_fast import load_data_fast
            train_reader, test_reader, val_reader, hyper_params = load_data_fast(hyper_params)
            print("Loaded preprocessed epoch files. Should be faster training...")
        except Exception as e:
            print("Tried loading preprocessed epoch files, but failed.")
            print("Please consider running `prep_all_data.sh` to make quick data for DeepCoNN/TransNet/NARRE.")
            print("This will save large amounts of run time.")
            print("Loading standard (slower) data..")
            train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    else:
        review_based_model = False
        train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)

    # Initialize the model
    model = Model(hyper_params)
    if is_cuda_available: model = model.cuda()
    xavier_init(model)

    # Train the model
    start_time = time.time()
    model = train_complete(
        hyper_params, Model, train_reader,
        val_reader, user_count, item_count, model, review=review_based_model
    )

    # Calculating RMSE on test-set
    # criterion = MSELoss()
    criterion = RMSELoss()
    metrics, user_count_mse_map, item_count_mse_map = evaluate(
        model, criterion, test_reader, hyper_params,
        user_count, item_count, review=review_based_model
    )

    # Calculating HR@1 on test-set
    _, test_reader2, _, _ = load_data(hyper_params)  # Needs default slow reader
    metrics.update(eval_ranking(model, test_reader2, hyper_params, review=review_based_model))

    log_end_epoch(hyper_params, metrics, 'final', time.time() - start_time, metrics_on='(TEST)')

    return metrics, user_count_mse_map, item_count_mse_map


def main(hyper_params, gpu_id=None):
    import torch

    # Setting GPU ID for running entire code ## Very Very Imp.
    if gpu_id is not None: torch.cuda.set_device(int(gpu_id))
    method = main_pytorch
    metrics, user_count_mse_map, item_count_mse_map = method(hyper_params, gpu_id=gpu_id)

    '''
    NOTE: In addition to metrics, we also provide the following for research purposes:
    - `user_count_mse_map`: 
        Python dict with key of type <int> and values as <list>
        where, 
            - Key: Test user's train-set frequency
            - Value: list containing MSE's for all test users with same train-frequency
    - `item_count_mse_map`: 
        Python dict with key of type <int> and values as <list>
        where, 
            - Key: Test item's train-set frequency
            - Value: list containing MSE's for all test items with same train-frequency
    '''

    return metrics


if __name__ == '__main__':
    from hyper_params import hyper_params

    main(hyper_params)
