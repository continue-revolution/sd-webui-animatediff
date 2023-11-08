
import pytest
import requests


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"


@pytest.fixture()
def simple_txt2img_request():
    return {
        "prompt": '1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece, BestQuality. UltraDetailed, <lora:yoimiya:0.8>, <lora:v2_lora_TiltDown:0.8>\n0: closed mouth\n8: open mouth,',
        "negative_prompt": "(sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes",
        "batch_size": 1,
        "steps": 2,
        "cfg_scale": 7,
        "alwayson_scripts": {
            'AnimateDiff': {
                'args': [{
                    'enable': True,
                    'batch_size': 2,
                    'video_length': 4,
                }]
            }
        }
    }


def test_txt2img_simple_performed(url_txt2img, simple_txt2img_request):
    '''
    This test checks the following:
    - simple t2v generation
    - prompt travel
    - infinite context generator
    - motion lora
    '''
    response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200
    assert isinstance(response.json()['images'][0], str)
