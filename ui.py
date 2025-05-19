import tkinter as tk
from tkinter import scrolledtext
from chat import ask_question_with_retriever, ask_question_direct
from loader import load_and_split_documents
from retriever import build_retriever

class ChatUI:
    def __init__(self, master):
        self.master = master
        self.master.title("🤖 Gemini Chat")
        self.master.geometry("600x700")
        self.master.configure(bg="#ECECEC")

        # Chat log hiển thị
        self.chat_log = scrolledtext.ScrolledText(
            master, state='disabled', wrap='word',
            font=("Segoe UI", 11), bg="#FFFFFF", fg="#000000",
            bd=0, relief="flat"
        )
        self.chat_log.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Khung nhập tin nhắn
        bottom_frame = tk.Frame(master, bg="#ECECEC")
        bottom_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.entry = tk.Entry(
            bottom_frame, font=("Segoe UI", 11), bg="#FFFFFF", fg="#000000",
            relief="solid", bd=1
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            bottom_frame, text="Gửi", font=("Segoe UI", 11, "bold"),
            bg="#4CAF50", fg="white", padx=10, pady=5,
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Nút chuyển đổi chế độ
        self.mode_button = tk.Button(
            master, text="Chuyển chế độ: RAG", font=("Segoe UI", 10, "bold"),
            bg="#2196F3", fg="white", padx=10, pady=5,
            command=self.toggle_mode
        )
        self.mode_button.pack(pady=(0, 10))

        # Load tài liệu và retriever
        self.retriever = None
        self.is_rag_mode = True
        self.load_retriever()

    def load_retriever(self):
        splits = load_and_split_documents("docs")
        if splits:
            self.retriever = build_retriever(splits)
            self.is_rag_mode = True
            self.mode_button.config(text="Chuyển chế độ: RAG")
            self.append_message("🤖", "🔍 Đã bật chế độ RAG với tài liệu.")
        else:
            self.retriever = None
            self.is_rag_mode = False
            self.mode_button.config(text="Chuyển chế độ: Gemini trực tiếp")
            self.append_message("🤖", "💡 Không có tài liệu. Sử dụng Gemini trực tiếp.")

    def toggle_mode(self):
        if self.is_rag_mode:
            # Chuyển sang Gemini trực tiếp
            self.retriever = None
            self.is_rag_mode = False
            self.mode_button.config(text="Chuyển chế độ: Gemini trực tiếp")
            self.append_message("🤖", "💡 Đã chuyển sang chế độ Gemini trực tiếp.")
        else:
            # Chuyển sang RAG (nếu có tài liệu)
            splits = load_and_split_documents("docs")
            if splits:
                self.retriever = build_retriever(splits)
                self.is_rag_mode = True
                self.mode_button.config(text="Chuyển chế độ: RAG")
                self.append_message("🤖", "🔍 Đã chuyển sang chế độ RAG với tài liệu.")
            else:
                self.append_message("🤖", "⚠️ Không có tài liệu để sử dụng RAG.")

    def append_message(self, sender, message):
        self.chat_log.config(state='normal')
        if sender == "🧑":
            self.chat_log.insert(tk.END, f"\n🧑 Bạn:\n", "user")
        else:
            self.chat_log.insert(tk.END, f"\n🤖 Gemini:\n", "bot")
        self.chat_log.insert(tk.END, f"{message}\n")
        self.chat_log.tag_config("user", foreground="#0B5394", font=("Segoe UI", 11, "bold"))
        self.chat_log.tag_config("bot", foreground="#38761D", font=("Segoe UI", 11, "bold"))
        self.chat_log.config(state='disabled')
        self.chat_log.yview(tk.END)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.append_message("🧑", user_input)
        self.entry.delete(0, tk.END)
        self.master.after(100, self.get_bot_response, user_input)

    def get_bot_response(self, user_input):
        if self.retriever:
            answer = ask_question_with_retriever(user_input, self.retriever)
        else:
            answer = ask_question_direct(user_input)
        self.append_message("🤖", answer)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatUI(root)
    root.mainloop()