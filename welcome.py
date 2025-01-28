import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

def on_start_click(root):
    print("Start button clicked!")
    # Utilise le chemin complet pour 'mynewprojects.py'
    subprocess.run(["python", "C:/Users/R.O/Desktop/Recherche/mynewprojects.py"], check=True)
    # Si tu veux garder la fenêtre principale ouverte, commente la ligne ci-dessous
    root.destroy()

def main():
    root = tk.Tk()
    root.title("EMSI School GUI")

    # Charger l'image
    emsi_logo = Image.open("C:/Users/R.O/Desktop/Recherche/emsi_logo.png")
    emsi_logo_photo = ImageTk.PhotoImage(emsi_logo)

    # Créer une étiquette et stocker une référence à l'image
    logo_label = tk.Label(root, image=emsi_logo_photo)
    # Garder la référence pour éviter la collecte des déchets
    logo_label.image = emsi_logo_photo  
    logo_label.pack(pady=10)

    # Bouton pour démarrer le programme
    start_button = ttk.Button(root, text="Start", command=lambda: on_start_click(root))
    start_button.pack(pady=10)

    # Boucle d'événements principale
    root.mainloop()

if __name__ == "__main__":  # Vérification correcte du main
    main()
