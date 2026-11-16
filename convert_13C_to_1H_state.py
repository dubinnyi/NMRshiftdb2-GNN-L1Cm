import torch
import argparse
import os
import pickle
import json
import nets
from i18n import _


def main():
    parser = argparse.ArgumentParser(
        description=_("Convert 13C state to initialize 1H model (Partial Transfer Learning)")
    )
    parser.add_argument("--state_13c", type=str, required=True,
                        help=_("Path to trained 13C .state file"))
    parser.add_argument("--meta_13c", type=str, required=True,
                        help=_("Path to 13C .meta or .json file (feature extractor architectures are identical)"))
    parser.add_argument("--output", type=str, required=True,
                        help=_("Output path for the new .state file ready for 1H training"))

    args = parser.parse_args()

    # 1. Load 13C metadata.
    # The graph feature extractor (gml) architecture and output dimension (OUT_DIM=1) are identical.
    print(_("Loading 13C metadata from {path}...", path=args.meta_13c))
    if args.meta_13c.endswith('.json'):
        with open(args.meta_13c, 'r') as f:
            meta = json.load(f)
    else:
        with open(args.meta_13c, 'rb') as f:
            meta = pickle.load(f)

    net_params = meta['net_params']

    # 2. Instantiate a fresh model with clean random weight/bias initialization.
    # This generates fresh weights for the final readout layers (lin_out) so predictions are not biased toward 13C shifts.
    net_new = nets.GraphVertModel(**net_params)
    state_new = net_new.state_dict()

    # 3. Load pre-trained 13C model weights.
    print(_("Loading 13C weights from {path}...", path=args.state_13c))
    state_13c = torch.load(args.state_13c, map_location='cpu', weights_only=True)

    # 4. Transfer ONLY the graph convolutional layers (gml.*) and input batch normalization parameters.
    transferred_count = 0
    for k in state_new.keys():
        if (k.startswith('gml.') or k.startswith('batch_norm.')) and k in state_13c:
            state_new[k] = state_13c[k]
            transferred_count += 1

    print(_("Successfully transferred graph feature extractor layers (gml.*): {count}", count=transferred_count))

    # 5. Save the combined state dictionary.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_new, args.output)
    print(_("Done! New state saved to: {path}", path=args.output))


if __name__ == "__main__":
    main()
