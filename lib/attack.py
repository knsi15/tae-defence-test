from lib.mdeattack_timeseries import main as run_de_attack

def generate_adversarial_examples(dataset, model_type):
    run_de_attack(dataset, model_type)
