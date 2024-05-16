python -m llava.serve.cli  \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "/home/caduser/HDD/vit_token_compress/PiToMe/output/horse_raw.png" \
    --algo $1 --ratio $2 
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    # --load-4bit