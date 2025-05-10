import re

class RoomRentalChatbot:
    def __init__(self):
        self.state = "GREETING"
        self.room_type = None
        self.duration = None
        self.total_cost = 0
        self.room_prices = {
            "single": 1200,
            "double": 800,
            "triple": 600
        }
    
    def _recognize_room_type(self, user_input):
        user_input = user_input.lower()
        single_pattern = r"\b(single|one|1|alone|solo)\b"
        double_pattern = r"\b(double|two|2|couple|pair)\b"
        triple_pattern = r"\b(triple|three|3|group)\b"
        if re.search(single_pattern, user_input):
            return "single"
        elif re.search(double_pattern, user_input):
            return "double"
        elif re.search(triple_pattern, user_input):
            return "triple"
        return None
    
    def _validate_duration(self, duration_str):
        if duration_str.isdigit():
            days = int(duration_str)
            return days if days > 0 else None
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        if duration_str.lower() in word_to_num:
            return word_to_num[duration_str.lower()]
        return None
    
    def _recognize_payment_method(self, user_input):
        user_input = user_input.lower()
        debit_pattern = r"\b(debit|bank card|visa|mastercard)\b"
        credit_pattern = r"\b(credit|card)\b"
        gpay_pattern = r"\b(gpay|google pay|upi|phonepe|paytm)\b"
        cancel_pattern = r"\b(cancel|stop|quit|exit)\b"
        if re.search(debit_pattern, user_input) and not re.search(credit_pattern, user_input):
            return "debit"
        elif re.search(credit_pattern, user_input):
            return "credit"
        elif re.search(gpay_pattern, user_input):
            return "gpay"
        elif re.search(cancel_pattern, user_input):
            return "cancel"
        return None
    
    def _handle_greeting_state(self):
        print("\nWelcome to Room Rental Chatbot!")
        print("We offer single, double, or triple sharing rooms.")
        self.state = "ROOM_TYPE"
        return "What type of room would you like? (single/double/triple)"
    
    def _handle_room_type_state(self, user_input):
        if user_input.lower() in ['exit', 'quit']:
            self.state = "EXIT"
            return "Goodbye! Thank you for visiting."
        self.room_type = self._recognize_room_type(user_input)
        if self.room_type:
            self.state = "DURATION"
            return f"Great! {self.room_type.capitalize()} room selected. How many days will you stay?"
        return "Sorry, I didn't understand. Please specify single, double, or triple room."
    
    def _handle_duration_state(self, user_input):
        if user_input.lower() in ['exit', 'quit']:
            self.state = "EXIT"
            return "Goodbye! Thank you for visiting."
        duration = self._validate_duration(user_input)
        if duration:
            self.duration = duration
            self.total_cost = self.duration * self.room_prices[self.room_type]
            self.state = "PAYMENT"
            summary = (f"\nBooking Summary:\n"
                    f"- Room type: {self.room_type}\n"
                    f"- Duration: {self.duration} days\n"
                    f"- Total cost: ₹{self.total_cost}\n\n"
                    "How would you like to pay? (debit/credit/gpay)")
            return summary
        return "Please enter a valid number of days (e.g., '3' or 'five')."
    
    def _handle_payment_state(self, user_input):
        if user_input.lower() in ['exit', 'quit']:
            self.state = "EXIT"
            return "Goodbye! Thank you for visiting."
        payment_method = self._recognize_payment_method(user_input)
        if payment_method == "cancel":
            self.state = "EXIT"
            return "Booking cancelled. Thank you!"
        elif payment_method:
            self.state = "COMPLETE"
            return (f"Payment successful via {payment_method}!\n"
                    f"Your {self.room_type} room for {self.duration} days is booked.\n"
                    "Thank you for using our service!")
        return "Please specify a valid payment method (debit/credit/gpay) or 'cancel'."
    
    def process_input(self, user_input):
        if self.state == "GREETING":
            return self._handle_greeting_state()
        elif self.state == "ROOM_TYPE":
            return self._handle_room_type_state(user_input)
        elif self.state == "DURATION":
            return self._handle_duration_state(user_input)
        elif self.state == "PAYMENT":
            return self._handle_payment_state(user_input)
        elif self.state == "COMPLETE":
            self.state = "EXIT"
            return "Session complete. Type anything to exit."
        else:
            return "Invalid state. Session ended."

if __name__ == "__main__":
    chatbot = RoomRentalChatbot()
    print("Type 'exit' at any time to quit.\n")
    response = chatbot.process_input("")
    while chatbot.state != "EXIT":
        print(response)
        user_input = input("You: ").strip()
        response = chatbot.process_input(user_input)
    print(response)