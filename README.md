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

$env:PYTHONPATH="src"
python -m autodev.cli loop `
  --project MonProjet `
  --objective-file "objectifs/hello.txt" `
  --patch-file "objectifs/patches_hello.txt" `
  --max-iter 3



$env:PYTHONPATH="src"
python -m autodev.cli auto `
  --project MonProjet `
  --objective-file "objectifs/hello.txt" `
  --patch-file "objectifs/patches_hello.txt"


$env:PYTHONPATH="src"
python -m autodev.cli loop `
  --project MonProjet `
  --objective-file "objectifs/hello.txt" `
  --patch-file "objectifs/patches_hello.txt" `
  --max-iter 5 `
  --no-stop-on-empty-todo
