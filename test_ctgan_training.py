import sys
sys.path.append('.')

from src.ai_engine.tabular_generator import TabularDataGenerator

def test_ctgan_training_losses():
    generator = TabularDataGenerator()
    # Assuming 'customers' is a valid table_name for training
    success = generator.train_model(table_name='customers', epochs=5)
    
    if success:
        model_info = generator.models['customers']
        print("\n--- CTGAN Training Losses ---")
        print("Generator Losses:", model_info['generator_losses'])
        print("Discriminator Losses:", model_info['discriminator_losses'])
    else:
        print("CTGAN model training failed.")

if __name__ == '__main__':
    test_ctgan_training_losses()
