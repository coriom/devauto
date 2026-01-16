$env:PYTHONPATH="src"
python -m autodev.cli loop `
  --project MonProjet `
  --objective-file "objectifs/hello.txt" `
  --max-iter 10


$env:PYTHONPATH="src"
python -m autodev.cli loop `
  --project MonProjet `
  --objective-file objectifs/hello.txt `
  --patch-file objectifs/patches_hello.txt `
  --max-iter 10
