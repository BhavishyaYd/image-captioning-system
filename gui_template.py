"""
Comprehensive GUI Template for Image Captioning + Gemini Chatbot System
Includes: Image Upload, Caption Generation, AI-Powered Chatbot, Multilingual Support
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from PIL import Image, ImageTk
import threading
import os
from dotenv import load_dotenv

# Import your project modules
from predict import generate_caption
from chatbot import GeminiChatbot

load_dotenv()


class ImageCaptioningChatbotGUI:
    """Complete GUI Template with Image Upload, Captioning, and Gemini Chatbot"""
    
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Mandarin": "zh-CN",
    }
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Captioning + AI Chatbot System")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # State variables
        self.current_image_path = None
        self.current_caption = ""
        self.current_photo = None
        self.chatbot = None
        self.is_generating = False
        self.conversation_history = []
        
        # Configure style
        self.root.configure(bg="#0f172a")
        self._configure_styles()
        
        # Build UI
        self._build_ui()
        
    def _configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors
        style.configure(
            "TButton",
            font=("Segoe UI", 10),
            padding=8,
        )
        style.configure(
            "TCombobox",
            font=("Segoe UI", 10),
            padding=6,
        )
        
    def _build_ui(self):
        """Build the complete UI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self._build_header(main_frame)
        
        # Control Panel
        self._build_control_panel(main_frame)
        
        # Content Area (Image + Chatbot)
        content_frame = tk.Frame(main_frame, bg="#0f172a")
        content_frame.pack(fill="both", expand=True, pady=10)
        
        # Left side - Image and Caption
        self._build_image_section(content_frame)
        
        # Right side - Chatbot
        self._build_chatbot_section(content_frame)
        
    def _build_header(self, parent):
        """Build header section"""
        header_frame = tk.Frame(parent, bg="#1e293b", relief="flat")
        header_frame.pack(fill="x", pady=(0, 10))
        
        title = tk.Label(
            header_frame,
            text="🖼️  Image Captioning + AI Chatbot System",
            font=("Segoe UI", 18, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=15,
            pady=12,
        )
        title.pack(anchor="w")
        
        subtitle = tk.Label(
            header_frame,
            text="Upload images, generate captions, and ask questions using Google Gemini AI",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#1e293b",
            padx=15,
            pady=(0, 8),
        )
        subtitle.pack(anchor="w")
        
    def _build_control_panel(self, parent):
        """Build control panel with buttons and dropdowns"""
        control_frame = tk.LabelFrame(
            parent,
            text="📋 Controls & Settings",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=10,
        )
        control_frame.pack(fill="x", pady=(0, 10))
        
        # Row 1: File operations
        row1 = tk.Frame(control_frame, bg="#1e293b")
        row1.pack(fill="x", pady=5)
        
        tk.Button(
            row1,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Segoe UI", 10, "bold"),
            bg="#38bdf8",
            fg="#0f172a",
            activebackground="#7dd3fc",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=5)
        
        tk.Button(
            row1,
            text="📷 Capture Webcam",
            command=self.capture_webcam,
            font=("Segoe UI", 10, "bold"),
            bg="#60a5fa",
            fg="#0f172a",
            activebackground="#93c5fd",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=5)
        
        tk.Button(
            row1,
            text="🔄 Clear Image",
            command=self.clear_image,
            font=("Segoe UI", 10, "bold"),
            bg="#ef4444",
            fg="#fff",
            activebackground="#f87171",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=5)
        
        # Row 2: Language and Chatbot initialization
        row2 = tk.Frame(control_frame, bg="#1e293b")
        row2.pack(fill="x", pady=5)
        
        tk.Label(
            row2,
            text="Language:",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0",
            bg="#1e293b",
        ).pack(side="left", padx=(5, 8))
        
        self.language_var = tk.StringVar(value="English")
        language_dropdown = ttk.Combobox(
            row2,
            textvariable=self.language_var,
            values=list(self.SUPPORTED_LANGUAGES.keys()),
            state="readonly",
            width=15,
            font=("Segoe UI", 10),
        )
        language_dropdown.pack(side="left", padx=5)
        
        tk.Button(
            row2,
            text="🤖 Initialize Chatbot",
            command=self.initialize_chatbot,
            font=("Segoe UI", 10, "bold"),
            bg="#8b5cf6",
            fg="#fff",
            activebackground="#a78bfa",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=5)
        
        tk.Button(
            row2,
            text="💬 Reset Conversation",
            command=self.reset_conversation,
            font=("Segoe UI", 10, "bold"),
            bg="#f59e0b",
            fg="#000",
            activebackground="#fbbf24",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=5)
        
    def _build_image_section(self, parent):
        """Build left image and caption section"""
        left_frame = tk.LabelFrame(
            parent,
            text="🖼️  Image & Caption",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=10,
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Image display
        self.image_label = tk.Label(
            left_frame,
            bg="#0f172a",
            fg="#64748b",
            text="📷 No image loaded\n\nClick 'Upload Image' or 'Capture Webcam'",
            font=("Segoe UI", 12),
            height=18,
            width=35,
            relief="sunken",
            bd=2,
        )
        self.image_label.pack(fill="both", expand=True, pady=(0, 10))
        
        # Generate Caption button
        tk.Button(
            left_frame,
            text="✨ Generate Caption",
            command=self.generate_caption,
            font=("Segoe UI", 11, "bold"),
            bg="#10b981",
            fg="#fff",
            activebackground="#34d399",
            relief="flat",
            padx=12,
            pady=8,
            width=40,
        ).pack(fill="x", pady=(0, 10))
        
        # Caption display
        caption_label = tk.Label(
            left_frame,
            text="Caption:",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        caption_label.pack(anchor="w", pady=(0, 5))
        
        self.caption_text = tk.Text(
            left_frame,
            font=("Segoe UI", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            height=5,
            width=45,
            relief="sunken",
            bd=2,
            wrap="word",
        )
        self.caption_text.pack(fill="both", expand=True)
        
        # Copy caption button
        tk.Button(
            left_frame,
            text="📋 Copy Caption",
            command=self.copy_caption,
            font=("Segoe UI", 9),
            bg="#64748b",
            fg="#fff",
            activebackground="#78909c",
            relief="flat",
            padx=10,
            pady=4,
        ).pack(fill="x", pady=(5, 0))
        
    def _build_chatbot_section(self, parent):
        """Build right chatbot conversation section"""
        right_frame = tk.LabelFrame(
            parent,
            text="💬 AI Chatbot Q&A",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=10,
        )
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Status indicator
        status_frame = tk.Frame(right_frame, bg="#1e293b")
        status_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            status_frame,
            text="Status:",
            font=("Segoe UI", 9, "bold"),
            fg="#e2e8f0",
            bg="#1e293b",
        ).pack(side="left", padx=(0, 8))
        
        self.status_indicator = tk.Label(
            status_frame,
            text="⚫ Not Initialized",
            font=("Segoe UI", 9),
            fg="#f59e0b",
            bg="#1e293b",
        )
        self.status_indicator.pack(side="left")
        
        # Conversation history display
        conversation_label = tk.Label(
            right_frame,
            text="Conversation History:",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        conversation_label.pack(anchor="w", pady=(0, 5))
        
        self.conversation_display = scrolledtext.ScrolledText(
            right_frame,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#e2e8f0",
            height=16,
            width=50,
            relief="sunken",
            bd=2,
            wrap="word",
        )
        self.conversation_display.pack(fill="both", expand=True, pady=(0, 10))
        
        # Configure text tags for styling
        self.conversation_display.tag_config("user", foreground="#38bdf8", font=("Segoe UI", 9, "bold"))
        self.conversation_display.tag_config("bot", foreground="#10b981", font=("Segoe UI", 9, "bold"))
        self.conversation_display.tag_config("timestamp", foreground="#64748b", font=("Segoe UI", 8))
        self.conversation_display.tag_config("error", foreground="#ef4444", font=("Segoe UI", 9, "bold"))
        
        # Input section
        input_label = tk.Label(
            right_frame,
            text="Your Question:",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        input_label.pack(anchor="w", pady=(0, 5))
        
        self.question_input = tk.Text(
            right_frame,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#e2e8f0",
            height=3,
            width=50,
            relief="sunken",
            bd=2,
            wrap="word",
        )
        self.question_input.pack(fill="x", pady=(0, 8))
        
        # Button frame
        button_frame = tk.Frame(right_frame, bg="#1e293b")
        button_frame.pack(fill="x")
        
        tk.Button(
            button_frame,
            text="🚀 Ask Question",
            command=self.ask_question,
            font=("Segoe UI", 10, "bold"),
            bg="#38bdf8",
            fg="#0f172a",
            activebackground="#7dd3fc",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(0, 5), fill="x", expand=True)
        
        tk.Button(
            button_frame,
            text="🗑️ Clear Chat",
            command=self.clear_conversation,
            font=("Segoe UI", 10, "bold"),
            bg="#ef4444",
            fg="#fff",
            activebackground="#f87171",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", fill="x", expand=True)
        
    # --- Event Handlers ---
    
    def upload_image(self):
        """Upload image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self._display_image(file_path)
            self.caption_text.delete("1.0", "end")
            self.current_caption = ""
            
    def capture_webcam(self):
        """Capture image from webcam"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save temporarily
                temp_path = Path("temp_webcam.jpg")
                cv2.imwrite(str(temp_path), frame)
                self.current_image_path = temp_path
                self._display_image(str(temp_path))
                self.caption_text.delete("1.0", "end")
                self.current_caption = ""
            else:
                messagebox.showerror("Error", "Failed to capture image")
        except Exception as e:
            messagebox.showerror("Error", f"Webcam error: {str(e)}")
            
    def clear_image(self):
        """Clear current image"""
        self.image_label.config(
            text="📷 No image loaded\n\nClick 'Upload Image' or 'Capture Webcam'"
        )
        self.current_image_path = None
        self.current_photo = None
        self.caption_text.delete("1.0", "end")
        self.current_caption = ""
        
    def _display_image(self, image_path):
        """Display image in label"""
        try:
            img = Image.open(image_path)
            # Resize to fit label
            img.thumbnail((500, 400), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_photo, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def generate_caption(self):
        """Generate caption for current image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        self.is_generating = True
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", "Generating caption...")
        
        def _generate():
            try:
                caption = generate_caption(str(self.current_image_path))
                self.current_caption = caption
                self.root.after(0, self._update_caption_display, caption)
            except Exception as e:
                self.root.after(0, self._show_error, f"Caption generation failed: {str(e)}")
            finally:
                self.is_generating = False
        
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        
    def _update_caption_display(self, caption):
        """Update caption text display"""
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", caption)
        
    def copy_caption(self):
        """Copy caption to clipboard"""
        if self.current_caption:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_caption)
            messagebox.showinfo("Success", "Caption copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "No caption to copy")
            
    def initialize_chatbot(self):
        """Initialize Gemini chatbot"""
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            messagebox.showerror("Error", "GOOGLE_API_KEY not found in .env file")
            return
        
        try:
            self.chatbot = GeminiChatbot(api_key=api_key)
            self.status_indicator.config(text="🟢 Ready", fg="#10b981")
            messagebox.showinfo("Success", "Chatbot initialized successfully!")
        except Exception as e:
            self.status_indicator.config(text="🔴 Error", fg="#ef4444")
            messagebox.showerror("Error", f"Failed to initialize chatbot: {str(e)}")
            
    def ask_question(self):
        """Ask question to chatbot"""
        if not self.chatbot:
            messagebox.showwarning("Warning", "Please initialize chatbot first")
            return
        
        if not self.current_caption:
            messagebox.showwarning("Warning", "Please generate a caption first")
            return
        
        question = self.question_input.get("1.0", "end").strip()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question")
            return
        
        # Set image context
        self.chatbot.set_image_context(self.current_caption)
        
        # Add user question to display
        self._add_to_conversation("You", question, "user")
        self.question_input.delete("1.0", "end")
        
        # Get response in thread
        def _get_response():
            try:
                selected_language = self.language_var.get()
                response = self.chatbot.ask(question, target_language=selected_language)
                self.root.after(0, self._add_to_conversation, "🤖 Bot", response, "bot")
            except Exception as e:
                self.root.after(0, self._add_to_conversation, "🤖 Bot", f"Error: {str(e)}", "error")
        
        thread = threading.Thread(target=_get_response, daemon=True)
        thread.start()
        
    def _add_to_conversation(self, speaker, message, tag=""):
        """Add message to conversation history"""
        self.conversation_display.config(state="normal")
        
        if self.conversation_display.get("1.0", "end").strip():
            self.conversation_display.insert("end", "\n" + "-" * 40 + "\n")
        
        self.conversation_display.insert("end", f"{speaker}: ", tag)
        self.conversation_display.insert("end", f"{message}\n")
        
        self.conversation_display.see("end")
        self.conversation_display.config(state="disabled")
        
    def reset_conversation(self):
        """Reset chatbot conversation"""
        if self.chatbot:
            self.chatbot.set_image_context(self.current_caption)
            self.clear_conversation()
            messagebox.showinfo("Success", "Conversation reset!")
        else:
            messagebox.showwarning("Warning", "Chatbot not initialized")
            
    def clear_conversation(self):
        """Clear conversation display"""
        self.conversation_display.config(state="normal")
        self.conversation_display.delete("1.0", "end")
        self.conversation_display.config(state="disabled")
        
    def _show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ImageCaptioningChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
