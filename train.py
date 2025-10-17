# train.py


from pathlib import Path
import torch
import torch.nn as nn
import argparse
from tqdm.auto import tqdm
import os.path as osp

from utils import transfer_to_gpu, get_jac_from_image_frame, safe_make_dirs
from datasets import get_dataset_class, dfn_sparse_batch_collate
from layers import LJN


# Loss weights
POS_LOSS_WEIGHT = 10.0
JAC_LOSS_2_WEIGHT = 0.5


def parse_args():
    argparser = argparse.ArgumentParser(description="Train LJN")
    argparser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment"
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="faust_o",
        choices=["faust_r", "faust_o", "surreal", "scape_r", "scape_o", "shrec19_r"],
        help="Dataset to use for training",
    )
    argparser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    argparser.add_argument("--n_epoch", type=int, default=120, help="Number of epochs")
    argparser.add_argument(
        "--chkpt", type=str, help="Path to a checkpoint to resume training from"
    )
    argparser.add_argument(
        "--only_eval", action="store_true", help="Only evaluate the model"
    )
    argparser.add_argument("--spec_inp", action="store_true", help="Use spectral input")
    argparser.add_argument(
        "--pos_loss_weight",
        type=float,
        default=10.0,
        help="Weight for the position loss",
    )
    argparser.add_argument(
        "--jac_loss_2_weight",
        type=float,
        default=0.5,
        help="Weight for the second jacobian loss",
    )

    return argparser.parse_args()


def train_one_epoch(
    model, train_dl, optimizer, criterion, pbar, pos_loss_weight, jac_loss_2_weight
):
    for batch_dict in tqdm(train_dl, leave=False):
        batch_dict = transfer_to_gpu(batch_dict)
        optimizer.zero_grad()

        pred_J, pred_pos = model(batch_dict)

        pos_loss = criterion(pred_pos.squeeze(), batch_dict["tar_v"].squeeze())

        gt_jac_proj = batch_dict["gt_jac"].squeeze() @ batch_dict[
            "src_f_basis"
        ].squeeze().transpose(2, 1)
        pred_jac_proj = pred_J @ batch_dict["src_f_basis"].squeeze().transpose(2, 1)
        jac_loss = torch.linalg.norm(
            pred_jac_proj - gt_jac_proj, "fro", dim=(-2, -1)
        ).mean()

        j_embed = get_jac_from_image_frame(pred_pos, batch_dict)
        jac_loss_2 = (
            (
                j_embed @ batch_dict["src_f_basis"].squeeze().transpose(2, 1)
                - gt_jac_proj
            )
            .norm("fro", dim=(-2, -1))
            .mean()
        )

        loss = pos_loss_weight * pos_loss + jac_loss + jac_loss_2_weight * jac_loss_2

        pbar.set_description(
            "Jac Loss: %.6f, Pos Loss: %.6f" % (jac_loss.item(), pos_loss.item())
        )
        loss.backward()
        optimizer.step()


def train(args, model, train_dl, optimizer, criterion):
    model.train()
    start_ep = 0
    if args.chkpt is not None:
        model.load_state_dict(torch.load(args.chkpt))
        print("Loaded checkpoint")
        start_ep = int(Path(args.chkpt).stem.split("_")[-1])
        print("Starting from epoch %d" % start_ep)

    save_dir = "./Logs/%s/" % args.exp_name
    safe_make_dirs(save_dir)

    pbar = tqdm(range(start_ep, args.n_epoch))
    for i in pbar:
        if i >= args.n_epoch // 2:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4

        train_one_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            pbar,
            args.pos_loss_weight,
            args.jac_loss_2_weight,
        )

        if (i + 1) % 5 == 0 and i > 0:
            torch.save(model.state_dict(), osp.join(save_dir, "pos_net_%d.pth" % i))
        torch.save(model.state_dict(), osp.join(save_dir, "latest.pth"))


def main():
    args = parse_args()
    model = LJN().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    if not args.only_eval:
        _, train_dl = get_dataset_class(
            args.dataset,
            batch_size=1,
            spec_inp=args.spec_inp,
            mode="train",
            collate_fn=dfn_sparse_batch_collate,
        )
        train(args, model, train_dl, optimizer, criterion)
    else:
        raise NotImplementedError(
            "Only training is supported for now. Eval functionality coming soon."
        )


if __name__ == "__main__":
    main()
