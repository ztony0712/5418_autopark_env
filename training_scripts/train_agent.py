# scripts/train_agent.py
import os
from autopark_env.agents.parking_agent import ParkingAgent

def main():
    # Instantiate ParkingAgent
    agent = ParkingAgent(env_name='my-new-env-v0')
    
    # Check if model files exist, if so, load the model
    if os.path.exists('saved_models/actor.pth') and os.path.exists('saved_models/critic.pth'):
        agent.load_model(path='saved_models/')
    else:
        print("No saved model found. Starting training from scratch.")
    
    # Start training
    agent.train(num_episodes=1000)  # Train for 100 episodes
    
    # Save the trained model
    agent.save_model(path='saved_models/')
    
    # Test the trained model
    print("Starting testing...")
    agent.test(num_episodes=5)  # Test for 5 episodes

if __name__ == "__main__":
    main()
