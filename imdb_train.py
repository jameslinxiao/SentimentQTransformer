import jax

from qtransformer.datasets import get_imdb_dataloaders
from qtransformer.training import train_and_evaluate
from qtransformer.transformers import Transformer
from qtransformer.quantum_layer import get_circuit

data_dir = './data'

for d in jax.devices():
    print(d, d.device_kind)

(imdb_train_dataloader, imdb_valid_dataloader, imdb_test_dataloader), vocab, tokenizer = get_imdb_dataloaders(batch_size=32, data_dir=data_dir, max_vocab_size=20_000, max_seq_len=512)
print(f"Vocabulary size: {len(vocab)}")
first_batch = next(iter(imdb_train_dataloader))
print(first_batch[0][0])
print(' '.join(map(bytes.decode, tokenizer.detokenize(first_batch[0])[0].numpy().tolist())))

#qtransformer
# model = Transformer(num_tokens=len(vocab), max_seq_len=512, num_classes=2, hidden_size=6, num_heads=2, num_transformer_blocks=4, mlp_hidden_size=3,
#                     quantum_attn_circuit=get_circuit(), quantum_mlp_circuit=get_circuit())
#transformer
model = Transformer(num_tokens=len(vocab), max_seq_len=512, num_classes=2, hidden_size=6, num_heads=2, num_transformer_blocks=4, mlp_hidden_size=3)

train_and_evaluate(model, imdb_train_dataloader, imdb_valid_dataloader, imdb_test_dataloader, num_classes=2, num_epochs=30)