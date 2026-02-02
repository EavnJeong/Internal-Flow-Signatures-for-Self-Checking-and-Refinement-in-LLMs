import os
import torch


def load_dataset(args, root):
    if args.dataset == 'halueval':
        from .halueval import HaluEvalDataset, collate_halueval
        if args.task == 'qa':
            data_path = os.path.join(root, "qa_data.json")
        elif args.task == 'general':
            data_path = os.path.join(root, "general_data.json")
        elif args.task == 'dialogue':
            data_path = os.path.join(root, "dialogue_data.json")
        elif args.task == 'summarization':
            data_path = os.path.join(root, "summarization_data.json")
        else:
            raise ValueError(f"Task {args.task} not recognized for HaluEval dataset.")

        dataset = HaluEvalDataset(
            data_path=data_path,
            task=args.task,
            mode="sample_one",
            seed=0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_halueval
        )
        return dataloader

    
    elif args.dataset == 'hotpot-wiki':
        from .hotpot import HotpotDataset
        from .collate_fns import collate_fn_hotpot
        dataset = HotpotDataset(os.path.join(root, "hotpot_dev_fullwiki_v1.json"))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_hotpot
        )
        return dataloader
    
    elif args.dataset == 'hotpot-distractor':
        from .hotpot import HotpotDataset
        from .hotpot import collate_fn_hotpot
        dataset = HotpotDataset(os.path.join(root, "hotpot_dev_distractor_v1.json"))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_hotpot
        )
        return dataloader

    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")