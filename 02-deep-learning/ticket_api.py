from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Support API")


class TicketCreate(BaseModel):
    title: str
    priority: int


tickets = [
    {
        "ticket_id": 101,
        "title": "Login Error",
        "priority": 1,
        "status": "open"
    },
    {
        "ticket_id": 102,
        "title": "Payment Error",
        "priority": 2,
        "status": "open"
    },
    {
        "ticket_id": 103,
        "title": "Network Issue",
        "priority": 2,
        "status": "closed"
    }
]


@app.get("/")
def home():
    return {
        "message": "Support API is running",
        "next_step": "Connect an AI model"
    }


@app.get("/ticket/{ticket_id}")
def get_ticket(ticket_id: int):
    for ticket in tickets:
        if ticket["ticket_id"] == ticket_id:
            return ticket

    return {"message": "Ticket not found"}


@app.get("/tickets")
def get_tickets(priority: int | None = None):
    if priority is None:
        return {"tickets": tickets}

    filtered_tickets = [
        ticket for ticket in tickets
        if ticket["priority"] == priority
    ]

    return {"tickets": filtered_tickets}


@app.post("/tickets", status_code=201)
def create_ticket(ticket_data: TicketCreate):
    new_ticket = {
        "ticket_id": max(
            (ticket["ticket_id"] for ticket in tickets),
            default=100
        ) + 1,
        "title": ticket_data.title,
        "priority": ticket_data.priority,
        "status": "open"
    }

    tickets.append(new_ticket)

    return new_ticket
