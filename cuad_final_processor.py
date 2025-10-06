# CUAD Dataset Processor - Perfect for your data format!
import json
import os
import pandas as pd
import numpy as np

def process_cuad_dataset():
    """
    Process the CUAD v1 dataset and prepare it for training
    """
    print("🚀 Processing CUAD Dataset for ML Training")
    print("=" * 50)
    
    # Load the dataset
    file_path = 'C:/Users/durga/kisan-service-portal-main/jurisai_assstant/jurisai_assstant/CUAD_v1/CUAD_v1.json'
    
    print("📂 Loading CUAD dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Dataset loaded: {len(data['data'])} items")
    
    # Extract contracts
    contracts = []
    total_paragraphs = 0
    
    for item in data['data']:
        title = item.get('title', 'Legal Contract')
        paragraphs = item.get('paragraphs', [])
        
        # Combine all paragraphs into one contract text
        contract_parts = []
        
        for paragraph in paragraphs:
            if 'context' in paragraph:
                contract_parts.append(paragraph['context'].strip())
        
        # Join all parts to make the full contract
        full_contract = '\n\n'.join(contract_parts)
        
        if len(full_contract) > 500:  # Only substantial contracts
            contracts.append({
                'title': title,
                'text': full_contract,
                'word_count': len(full_contract.split()),
                'paragraph_count': len(contract_parts)
            })
            total_paragraphs += len(contract_parts)
    
    print(f"📄 Extracted {len(contracts)} contracts")
    print(f"📊 Total paragraphs processed: {total_paragraphs}")
    
    # Calculate statistics
    word_counts = [c['word_count'] for c in contracts]
    
    print(f"\n📊 Contract Statistics:")
    print(f"   - Total contracts: {len(contracts)}")
    print(f"   - Average length: {np.mean(word_counts):.0f} words")
    print(f"   - Median length: {np.median(word_counts):.0f} words")
    print(f"   - Min length: {min(word_counts)} words")
    print(f"   - Max length: {max(word_counts)} words")
    
    # Show sample
    if contracts:
        sample = contracts[0]
        print(f"\n📋 Sample Contract:")
        print(f"   Title: {sample['title']}")
        print(f"   Word count: {sample['word_count']}")
        print(f"   Paragraphs: {sample['paragraph_count']}")
        print(f"   Text preview: {sample['text'][:300]}...")
    
    # Create directories
    os.makedirs('datasets', exist_ok=True)
    
    # Save processed data
    df = pd.DataFrame(contracts)
    df.to_csv('datasets/processed_contracts.csv', index=False)
    print(f"\n✅ Processed data saved: datasets/processed_contracts.csv")
    
    # Create training data
    print(f"\n🔧 Creating training data for ML model...")
    
    training_texts = []
    
    for contract in contracts:
        if contract['word_count'] >= 200:  # Filter minimum length
            # Add special tokens for training
            formatted_text = f"<|startoftext|>{contract['text']}<|endoftext|>"
            training_texts.append(formatted_text)
    
    # Save training data
    training_file = 'datasets/training_data.txt'
    with open(training_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(training_texts))
    
    file_size_mb = os.path.getsize(training_file) / 1024 / 1024
    
    print(f"✅ Training data created:")
    print(f"   - File: {training_file}")
    print(f"   - Contracts: {len(training_texts)}")
    print(f"   - File size: {file_size_mb:.1f} MB")
    
    print(f"\n🎯 Ready for Model Training!")
    print(f"   - Next step: Run the model training script")
    print(f"   - Expected training time: 2-4 hours")
    
    return df

def create_sample_documents():
    """
    Create some sample legal documents for testing
    """
    print("\n🧪 Creating sample documents for testing...")
    
    samples = [
        {
            'type': 'employment_contract',
            'inputs': {
                'employer': 'TechCorp Solutions Pvt. Ltd.',
                'employee': 'Rajesh Kumar',
                'position': 'Software Engineer',
                'salary': '₹85,000'
            }
        },
        {
            'type': 'rental_agreement', 
            'inputs': {
                'landlord': 'Mumbai Properties Ltd.',
                'tenant': 'Priya Sharma',
                'address': '123 Marine Drive, Mumbai',
                'rent': '₹35,000'
            }
        }
    ]
    
    # Save sample data
    with open('datasets/sample_inputs.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("✅ Sample test data created: datasets/sample_inputs.json")

if __name__ == "__main__":
    # Process the main dataset
    df = process_cuad_dataset()
    
    if df is not None and len(df) > 0:
        # Create sample documents
        create_sample_documents()
        
        print("\n🎉 COMPLETE SUCCESS!")
        print("=" * 50)
        print("✅ CUAD dataset processed")
        print("✅ Training data prepared") 
        print("✅ Sample test data created")
        print("\n🔄 Next steps:")
        print("1. Install PyTorch: pip install torch transformers")
        print("2. Run model training script")
        print("3. Test with sample documents")
        print("\n🚀 Your VakilSeva will have a custom-trained legal document generator!")
        
    else:
        print("\n❌ Processing failed!")