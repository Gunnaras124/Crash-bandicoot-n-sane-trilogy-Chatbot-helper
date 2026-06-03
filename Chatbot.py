from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import scrolledtext


# Load sentence similarity model (downloads ~80MB on first run)
model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

qa_pairs = [
    ("hello", "Woah! Ready to spin into some Crash Bandicoot action?"),
    ("hi", "Woah! Ready to spin into some Crash Bandicoot action?"),
    ("hey", "Woah! Ready to spin into some Crash Bandicoot action?"),

    ("trilogy", "Crash Bandicoot N. Sane Trilogy includes Crash 1, 2, and Warped!"),

    ("crash1 1", "Crash 1 is the hardest—no saves unless you earn them!"),
    ("crash2 2", "Crash 2 adds warp rooms and smoother gameplay!"),
    ("crash3 3", "Crash 3 introduces abilities like double jump and bazooka!"),

    ("mask aku masks akuaku", "Masks give protection, and 3 Aku Aku masks grant temporary invincibility!"),

    ("neocortex cortex drcortex drneocortex", "Dr. Neo Cortex is the main villain trying to control the islands!"),
    ("brio nbrio drnbrio drbrio", "Dr. N. Brio is Cortex’s assistant who mutates himself into a monster!"),
    ("tiny tinytiger", "Tiny Tiger is a strong boss who chases you in arena fights!"),
    ("dingodile", "Dingodile uses a flamethrower in his boss battle!"),
    ("ngin drngin", "N. Gin has a rocket in his head and attacks with missiles!"),
    ("tropy ntropy drtropy drntropy", "N. Tropy is the time master you face in time-themed boss fights!"),

    ("gems gem diamond diamonds", "Collect all crates in a level to earn a gem!"),
    ("relics relic timetrial timetrials time trial trials", "Relics are earned by completing time trials quickly!"),
    ("crates boxes box crate", "Break every crate in a level to get 100% completion!"),

    ("wumpa", "Wumpa fruit act like coins—collect 100 to earn an extra life!"),
    ("lives", "Lives let you retry levels—be careful not to lose them all!"),

    ("coco", "Coco is Crash’s sister and a playable character in most levels!"),
    ("polar", "Polar is the cute bear cub you ride in snowy levels!"),
    ("pura", "Pura is the tiger cub you ride in Crash 3!"),

    ("warp room warproom", "Warp rooms let you select and replay levels in Crash 2 and 3!"),
    ("portal", "Portals transport you into different levels from the hub area!"),

    ("bonus round bonusround", "Bonus rounds let you earn extra lives and practice without risk!"),
    ("secret", "Some levels have hidden paths that unlock secret stages!"),

    ("spin", "Spin attack is Crash’s main move for defeating enemies and breaking crates!"),
    ("jump", "Precise jumping is key to surviving tricky platform sections!"),

    ("tips", "Take it slow and focus on timing your jumps carefully!"),

    ("lore", "Crash was created by Cortex but escaped and turned against him!"),
    ("islands", "The games take place across the Wumpa Islands!"),

    ("thanks", "No worries! Keep spinning and smashing those crates!"),
    ("bye", "See ya! And remember… spin to win!"),
]
question_texts=[q for q, a in qa_pairs]
question_embeddings=model.encode(question_texts, convert_to_tensor=True)

THRESHOLD=0.3

def get_response(user_input):
    input_embedding=model.encode(user_input, convert_to_tensor=True)
    similarities=util.cos_sim(input_embedding, question_embeddings)[0]
    best_idx=similarities.argmax().item()
    best_score=similarities[best_idx].item()

    if best_score<THRESHOLD:
        return best_score, "Sorry, i don't understand. Try asking about wumpa, masks, dingodile etc"

    return best_score, qa_pairs[best_idx][1]
class ChatbotUI:
    def __init__(self,root):
        self.root=root
        self.root.title("Crash bandicoot n sane trilogy Chatbot")
        self.root.geometry("500x600")
        self.root.configure(bg="#2E2E2E")

        tk.Label(
            root,text="Crash bandicoot n sane trilogy Chatbot",
            font=("Helvetica",16,"bold"),
            fg="#FFFFFF",bg="#2E2E2E"
        ).pack(pady=10)
        self.chat_area=scrolledtext.ScrolledText(
            root,wrap=tk.WORD,height=20,width=50,font=("Arial",11),
            bg="#3C3C3C",fg="#E0E0E0",insertbackground="white"
        )
        self.chat_area.pack(pady=10,padx=10)
        self.chat_area.insert(tk.END,
                              "Welcome to the crash bandicoot n sane trilogy Chatbot!\n"
                              "Ask about crash bandicoot n sane trilogy(e.g. boxes, gems, coco, ngin e.t.c.")
        self.chat_area.config(state='disabled')

        input_frame=tk.Frame(root, bg="#2E2E2E")
        input_frame.pack(pady=5)

        self.input_field=tk.Entry(
            input_frame, width=40, font=("Arial",11), bg="#4A4A4A", fg="#FFFFFF",
            insertbackground="white"
        )
        self.input_field.pack(side=tk.LEFT, padx=5)
        self.input_field.bind("<Return>", self.send_message)

        tk.Button(
            input_frame, text="Send", command=self.send_message, font=("Arial", 11),
            bg="#4CAF50", fg="#FFFFFF", activebackground="#45A049"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            root, text="Clear Chat", command=self.clear_chat, font=("Arial", 11),
            bg="#F44336", fg="#FFFFFF", activebackground="#032F2F"
        ).pack(pady=5)

    def send_message(self, event=None):
        user_input=self.input_field.get().strip()
        if not user_input:
            return

        score, response=get_response(user_input)
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"\nYou: {user_input}\n")
        self.chat_area.insert(tk.END, f"Match confidence: {score:.2f}\n")
        self.chat_area.insert(tk.END, f"Bot: {response}\n")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        self.input_field.delete(0, tk.END)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0,tk.END)
        self.chat_area.insert(tk.END,
                              "Welcome to the Crash Bandicoot n sane trilogy Chatbot!!!!\n"
                              "Ask about crash bandicoot n sane trilogy for things like gems, lives, ngin etc.\n")
        self.chat_area.config(state='disabled')
        
def main():
    root=tk.Tk()
    app=ChatbotUI(root)
    root.mainloop()

if __name__=="__main__":
    main()