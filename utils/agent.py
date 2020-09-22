import torch
import utils

from model import QModel


class Agent:
    """
        An agent.

        It is able:
        - to choose an action given an observation,
        - to analyze the feedback (i.e. reward and done state) of its action.
    """

    def __init__(self, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(
            obs_space
        )
        self.model = QModel(obs_space, action_space)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        self.model.load_state_dict(utils.get_model_state(model_dir))
        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        Q = self.model(preprocessed_obss)
        actions = torch.argmax(Q, dim=1, keepdim=True)

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        pass

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

