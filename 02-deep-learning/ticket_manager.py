
# Store ticket information
ticket = {
    "id": 101,
    "title": "Login Error",
    "priority": 3,
    "status": "open"
}

# Read ticket information
title = ticket["title"]
priority = ticket["priority"]

# Generate a message based on priority

def get_priority_message(priority):
    if priority == 1:
        return "Critical ticket: handle immediately."
    elif priority == 3:
            return "Normal priority ticket."
    elif priority == 2:
        return "High priority ticket: handle soon."
    else:
        return "Normal priority ticket."
    


message = get_priority_message(ticket["priority"])
print("Message:", message)

# Display ticket information
print("Ticket ID:", ticket["id"])
print("Title:", title)
print("Priority:", priority)
print("Status:", ticket["status"])
print("Message:", message)