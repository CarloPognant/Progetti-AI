#!/usr/bin/env python3
"""
Main entry point - Snake AI
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════╗
║              🐍 SNAKE AI - Training & Testing              ║
╚════════════════════════════════════════════════════════════╝

Uso:
  python main.py train     - Allena il modello
  python main.py test      - Testa il modello
  python main.py clean     - Cancella i modelli
  python main.py info      - Mostra informazioni

Esempio:
  python main.py train
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        print("🚀 Avvio training...")
        os.system("python src/train.py")
    
    elif command == 'test':
        print("🎮 Avvio test...")
        os.system("python src/test_ai.py")
    
    elif command == 'info':
        from config import MODEL_BEST_PATH, BEST_SCORE_PATH
        print("\n" + "=" * 60)
        print("📊 INFORMAZIONI")
        print("=" * 60)
        print(f"📁 Modelli: models/")
        print(f"📁 Log: logs/")
        print(f"📁 Codice: src/")
        if os.path.exists(MODEL_BEST_PATH):
            print(f"\n✓ Modello trovato ({os.path.getsize(MODEL_BEST_PATH)/1024:.1f} KB)")
        else:
            print(f"\n✗ Nessun modello (esegui training)")
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH) as f:
                score = f.read().strip()
            print(f"✓ Best score: {score} mele")
        print("=" * 60 + "\n")
    
    elif command == 'clean':
        import shutil
        if os.path.exists("models"):
            shutil.rmtree("models")
            print("✓ Modelli cancellati")
    
    else:
        print(f"❌ Comando sconosciuto: {command}")

if __name__ == '__main__':
    main()