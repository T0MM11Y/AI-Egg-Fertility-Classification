import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Application theme colors
DARK_BG = "#121212"
CARD_BG = "#1E1E1E"
ACCENT_COLOR = "#2C6BED"
SUCCESS_COLOR = "#4CAF50"
DANGER_COLOR = "#F44336"
TEXT_COLOR = "#FFFFFF"
SECONDARY_TEXT = "#A0A0A0"

class EggFertilityClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Egg Fertility Classifier")
        self.root.geometry("500x700")
        self.root.configure(bg=DARK_BG)
        self.root.resizable(False, False)
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model("best_model.h5")
            self.model_loaded = True
        except:
            print("Warning: Model not found. Running in demo mode.")
            self.model_loaded = False
        
        # App icon
        try:
            icon_path = "app_icon.ico"
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
            
        self.setup_ui()
        
    def setup_ui(self):
        # Header with logo
        header_frame = Frame(self.root, bg=DARK_BG, height=80)
        header_frame.pack(fill="x", pady=(20, 10))
        
        logo_label = Label(header_frame, text="🥚", font=("Arial", 30), bg=DARK_BG, fg=TEXT_COLOR)
        logo_label.pack(side="left", padx=(30, 10))
        
        title_frame = Frame(header_frame, bg=DARK_BG)
        title_frame.pack(side="left")
        
        app_title = Label(title_frame, text="Egg Fertility Classifier", 
                         font=("Segoe UI", 18, "bold"), bg=DARK_BG, fg=TEXT_COLOR)
        app_title.pack(anchor="w")
        
        app_subtitle = Label(title_frame, text="AI-powered egg quality analysis", 
                            font=("Segoe UI", 10), bg=DARK_BG, fg=SECONDARY_TEXT)
        app_subtitle.pack(anchor="w")
        
        # Main content area
        main_frame = Frame(self.root, bg=CARD_BG, padx=25, pady=25, bd=0)
        main_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Image preview area with border
        self.preview_frame = Frame(main_frame, bg=CARD_BG, bd=0)
        self.preview_frame.pack(pady=(0, 20))
        
        self.preview_canvas = Canvas(self.preview_frame, width=300, height=300, 
                                    bg="#242424", bd=0, highlightthickness=0)
        self.preview_canvas.pack()
        
        # Image placeholder text
        self.placeholder_text = self.preview_canvas.create_text(
            150, 150, text="Image Preview", fill=SECONDARY_TEXT, font=("Segoe UI", 12))
            
        # Image display label
        self.img_label = Label(self.preview_canvas, bg="#242424")
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Result panel
        result_frame = Frame(main_frame, bg=CARD_BG, padx=10, pady=10)
        result_frame.pack(fill="x")
        
        # Result labels
        self.result_title = Label(result_frame, text="Classification Result", 
                                font=("Segoe UI", 14, "bold"), bg=CARD_BG, fg=TEXT_COLOR)
        self.result_title.pack(anchor="w", pady=(0, 10))
        
        # Result status
        self.result_frame = Frame(result_frame, bg=CARD_BG)
        self.result_frame.pack(fill="x", pady=5)
        
        self.result_label = Label(self.result_frame, text="Not analyzed yet", 
                                font=("Segoe UI", 16, "bold"), bg=CARD_BG, fg=TEXT_COLOR)
        self.result_label.pack(side="left")
        
        # Confidence score
        self.confidence_frame = Frame(result_frame, bg=CARD_BG)
        self.confidence_frame.pack(fill="x", pady=(5, 15))
        
        self.confidence_label = Label(self.confidence_frame, text="Confidence: --", 
                                    font=("Segoe UI", 12), bg=CARD_BG, fg=SECONDARY_TEXT)
        self.confidence_label.pack(side="left")
        
        # Buttons
        button_frame = Frame(main_frame, bg=CARD_BG)
        button_frame.pack(fill="x", pady=(20, 0))
        
        self.select_btn = Button(
            button_frame, 
            text="SELECT IMAGE", 
            command=self.select_and_classify_image,
            font=("Segoe UI", 11, "bold"), 
            bg=ACCENT_COLOR, 
            fg=TEXT_COLOR,
            activebackground=ACCENT_COLOR,
            activeforeground=TEXT_COLOR,
            bd=0,
            padx=15,
            pady=10,
            width=15,
            cursor="hand2"
        )
        self.select_btn.pack(side="left", padx=(0, 10))
        
        self.reset_btn = Button(
            button_frame, 
            text="RESET", 
            command=self.reset_analysis,
            font=("Segoe UI", 11), 
            bg="#3A3A3A", 
            fg=TEXT_COLOR,
            activebackground="#303030",
            activeforeground=TEXT_COLOR,
            bd=0,
            padx=15,
            pady=10,
            width=8,
            cursor="hand2"
        )
        self.reset_btn.pack(side="left")
        
        # Status message
        self.status_label = Label(self.root, text="Ready to analyze", 
                                font=("Segoe UI", 9), bg=DARK_BG, fg=SECONDARY_TEXT)
        self.status_label.pack(pady=(0, 5))
        
        # Footer
        footer_text = "Demo Mode" if not self.model_loaded else "© 2023 Egg Fertility Classifier v1.0"
        self.footer_label = Label(self.root, text=footer_text, 
                                font=("Segoe UI", 8), bg=DARK_BG, fg=SECONDARY_TEXT)
        self.footer_label.pack(pady=(0, 15))
    
    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        try:
            img = Image.open(image_path).resize((150, 150))  # Resize to model's input size
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            return None
    
    def select_and_classify_image(self):
        """Select an image and classify it"""
        file_path = filedialog.askopenfilename(
            title="Select Egg Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        
        if not file_path:
            return
            
        try:
            # Update status
            self.status_label.config(text="Analyzing image...")
            
            # Display the selected image
            self.display_image(file_path)
            
            # Perform classification
            if self.model_loaded:
                img_array = self.preprocess_image(file_path)
                if img_array is not None:
                    prediction = self.model.predict(img_array)
                    probability = prediction[0][0]
                    self.show_result(probability)
            else:
                # Demo mode - random prediction
                import random
                probability = random.uniform(0.1, 0.9)
                self.show_result(probability)
                self.status_label.config(text="Demo mode: Using random prediction")
                
        except Exception as e:
            self.status_label.config(text=f"Error during analysis: {str(e)}")
    
    def display_image(self, file_path):
        """Display the selected image in the preview area"""
        # Hide placeholder text
        self.preview_canvas.itemconfig(self.placeholder_text, state="hidden")
        
        # Open and resize image for display
        img = Image.open(file_path)
        
        # Calculate new dimensions while maintaining aspect ratio
        width, height = img.size
        max_size = 280  # Max dimension
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
            
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.tk_img = ImageTk.PhotoImage(img)
        
        # Update image label
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img  # Keep a reference
    
    def show_result(self, probability):
        """Display classification results"""
        is_fertile = probability < 0.5  # Assuming >0.5 means infertile
        
        result_text = "Fertile" if is_fertile else "Infertile"
        result_color = SUCCESS_COLOR if is_fertile else DANGER_COLOR
        
        # Update result labels
        self.result_label.config(text=result_text, fg=result_color)
        self.confidence_label.config(text=f"Confidence: {probability:.2%}")
        
        # Update status
        self.status_label.config(text="Analysis complete")
    
    def reset_analysis(self):
        """Reset the UI to initial state"""
        # Clear image
        self.img_label.config(image="")
        
        # Show placeholder
        self.preview_canvas.itemconfig(self.placeholder_text, state="normal")
        
        # Reset result labels
        self.result_label.config(text="Not analyzed yet", fg=TEXT_COLOR)
        self.confidence_label.config(text="Confidence: --")
        
        # Reset status
        self.status_label.config(text="Ready to analyze")


if __name__ == "__main__":
    root = tk.Tk()
    app = EggFertilityClassifier(root)
    root.mainloop()