import tkinter as tk
from chat import ask_question_with_retriever, ask_question_direct
from loader import load_and_split_documents
from retriever import build_retriever

class ChatUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gemini Chat")
        self.master.geometry("500x600")

        self.chat_log = tk.Text(master, state='disabled', wrap='word', bg="#F0F0F0")
        self.chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry = tk.Entry(master, width=80)
        self.entry.pack(padx=10, pady=(0, 10), side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(master, text="Gửi", command=self.send_message)
        self.send_button.pack(padx=(0, 10), pady=(0, 10), side=tk.RIGHT)

        # Load tài liệu và retriever
        self.retriever = None
        self.load_retriever()

    def load_retriever(self):
        splits = load_and_split_documents("docs")
        if splits:
            self.retriever = build_retriever(splits)
            self.append_message("🤖", "Đã bật chế độ RAG với tài liệu.")
        else:
            self.append_message("🤖", "Không tìm thấy tài liệu. Chat trực tiếp với Gemini.")

    def append_message(self, sender, message):
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, f"{sender}: {message}\n")
        self.chat_log.config(state='disabled')
        self.chat_log.see(tk.END)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.append_message("🧑", user_input)
        self.entry.delete(0, tk.END)

        if self.retriever:
            answer = ask_question_with_retriever(user_input, self.retriever)
        else:
            answer = ask_question_direct(user_input)

        self.append_message("🤖", answer)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatUI(root)
    root.mainloop()
