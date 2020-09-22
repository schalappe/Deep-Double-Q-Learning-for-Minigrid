import sys
import time
import utils
import torch
import argparse
import datetime
import numpy as np
import tensorboardX

from model import QModel
from module import QLearn


# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=5,
                    help="number of updates between two logs (default: 10)")
parser.add_argument("--save-interval", type=int, default=25,
                    help="number of updates between two saves (default: 50, \
                        0 means no saving)")
parser.add_argument("--update-interval", type=int, default=100,
                    help="update frequece of target network (default: 1000)")
parser.add_argument("--frames", type=int, default=10**6,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--max-memory", type=int, default=100000,
                    help="Maximum experiences stored (default: 100000)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")

args = parser.parse_args()

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_q_learn_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environment
env = utils.make_env(args.env, args.seed)
txt_logger.info("Environments loaded\n")

# Load training status
try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(
    env.observation_space
)

if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
policy_network = QModel(obs_space, env.action_space)
target_network = QModel(obs_space, env.action_space)
if "model_state" in status:
    policy_network.load_state_dict(status["model_state"])
target_network.load_state_dict(policy_network.state_dict())
policy_network.to(device)
target_network.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(policy_network))

algo = QLearn(env, policy_network, target_network, device, args.max_memory,
              args.discount, args.lr, args.update_interval, args.batch_size,
              preprocess_obss)

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters
    update_start_time = time.time()
    logs = algo.collect_experiences()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["rewards"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()
        header += ["policy_loss"]
        data += [np.mean(logs["loss"])]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | pL {:.3f}"
            .format(*data)
        )

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": policy_network.state_dict(),
                  "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
