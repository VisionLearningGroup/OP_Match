import logging
from utils import test, test_ood

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0
def eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, ema_model):
    if args.amp:
        from apex import amp
    global best_acc
    global best_acc_val

    model.eval()
    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model
    epoch = 0
    if args.local_rank in [-1, 0]:
        val_acc = test(args, val_loader, test_model, epoch, val=True)
        test_loss, close_valid, test_overall, \
        test_unk, test_roc, test_roc_softm, test_id \
            = test(args, test_loader, test_model, epoch)
        for ood in ood_loaders.keys():
            roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
            logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

        overall_valid = test_overall
        unk_valid = test_unk
        roc_valid = test_roc
        roc_softm_valid = test_roc_softm
        logger.info('validation closed acc: {:.3f}'.format(val_acc))
        logger.info('test closed acc: {:.3f}'.format(close_valid))
        logger.info('test overall acc: {:.3f}'.format(overall_valid))
        logger.info('test unk acc: {:.3f}'.format(unk_valid))
        logger.info('test roc: {:.3f}'.format(roc_valid))
        logger.info('test roc soft: {:.3f}'.format(roc_softm_valid))
    if args.local_rank in [-1, 0]:
        args.writer.close()
