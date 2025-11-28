
import random

# Simple Rock-Paper-Scissors game logic
# choices: 1 -> Rock, 2 -> Paper, 3 -> Scissors

def Game(player_choice, player_score, computer_score):
    choices = {1: 'Rock', 2: 'Paper', 3: 'Scissor'}
    computer = random.randint(1, 3)
    status = ''
    # tie
    if player_choice == computer:
        status = f"Tie: both {choices[player_choice]}"
    else:
        # player win cases
        wins = {1: 3, 2: 1, 3: 2}  # key beats value (e.g., 1 (rock) beats 3 (scissor))
        if wins[player_choice] == computer:
            player_score += 1
            status = f"Player wins: {choices[player_choice]} beats {choices[computer]}"
        else:
            computer_score += 1
            status = f"Computer wins: {choices[computer]} beats {choices[player_choice]}"
    return status, player_score, computer_score, computer
