python LLaVA/llava/serve/cli.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "/home/caduser/HDD/vit_token_compress/PiToMe/examples/images/Confusing-Pictures.jpg" \
    --algo $1 --ratio $2 
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    # --load-4bit