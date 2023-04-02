from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import AudioSerializer
from django.core.files.storage import FileSystemStorage
import random
from time import sleep


import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import Levenshtein as Lev
from torch import Tensor

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def levenshtein(s1, s2, cost=None, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug=debug)

    if len(s2) == 0:
        return len(s1)

    if cost is None:
        cost = {}

    # changed
    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]

def compose(chosung, jungsung, jongsung):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char

def decompose(c):
    if not character_is_korean(c):
        return None
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')

    # decomposition rule
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )    
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))


def jamo_levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return levenshtein(decompose(c1), decompose(c2))/3

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(['%.3f'%v for v in current_row[1:]])

        previous_row = current_row

    return previous_row[-1]


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


# parser = argparse.ArgumentParser(description='KoSpeech')
# parser.add_argument('--model_path', type=str, required=False, default='./model/model.pt')
# parser.add_argument('--audio_path', type=str, required=False, default='../media/record.wav')
# parser.add_argument('--device', type=str, required=False, default='cpu')
# opt = parser.parse_args()

# opt = {''}


word_lst =["사과"
,"아보카도"
,"바나나"
,"피망"
,"브로콜리"
,"양배추"
,"당근"
,"자두"
,"옥수수"
,"오이"
,"가지"
,"마늘"
,"생강"
,"포도"
,"청포도"
,"대파"
,"대추"
,"키위"
,"레몬"
,"상추"
,"귤"
,"망고"
,"멜론"
,"버섯"
,"양배추"
,"양파"
,"오렌지"
,"참외"
,"파프리카"
,"완두콩"
,"복숭아"
,"배"
,"고추"
,"감"
,"파인애플"
,"감자"
,"호박"
,"무"
,"시금치"
,"딸기"
,"고구마"
,"토마토"
,"수박"
,"배우"
,"건축가"
,"군인"
,"화가"
,"우주비행사"
,"작가"
,"농구선수"
,"경호원"
,"촬영기사"
,"카레이서"
,"목수"
,"치어리더"
,"요리사"
,"환경미화원"
,"작곡가"
,"지휘자"
,"상담사"
,"댄서"
,"집배원"
,"개발자"
,"영화감독"
,"의사"
,"농부"
,"소방관"
,"어부"
,"미용사"
,"헬스트레이너"
,"야구선수"
,"기자"
,"판사"
,"마술사"
,"마사지사"
,"정비사"
,"광부"
,"스님"
,"수녀"
,"간호사"
,"피아니스트"
,"조종사"
,"경찰"
,"프로게이머"
,"과학자"
,"판매원"
,"가수"
,"축구선수"
,"승무원"
,"학생"
,"선생님"
,"웨이터"
,"유튜버"
,"악어"
,"곰"
,"벌"
,"나비"
,"카멜레온"
,"고양이"
,"닭"
,"치타"
,"소"
,"게"
,"달팽이"
,"사슴"
,"두더지"
,"개"
,"오리"
,"코끼리"
,"물고기"
,"여우"
,"개구리"
,"기린"
,"해파리"
,"하마"
,"하이에나"
,"말"
,"잠자리"
,"캥거루"
,"사자"
,"병아리"
,"미어캣"
,"메뚜기"
,"원숭이"
,"쥐"
,"무당벌레"
,"너구리"
,"문어"
,"오랑우탄"
,"올빼미"
,"팬더"
,"돼지"
,"풍뎅이"
,"토끼"
,"사마귀"
,"상어"
,"양"
,"뱀"
,"호랑이"
,"거북이"
,"고래"
,"늑대"
,"염소"
,"에어컨"
,"자동차"
,"백팩"
,"바구니"
,"자전거"
,"칠판"
,"카메라"
,"카드"
,"의자"
,"바지"
,"노트북"
,"컵"
,"커튼"
,"책상"
,"문"
,"이어폰"
,"엘리베이터"
,"지우개"
,"비상문"
,"안경"
,"꽃"
,"냉장고"
,"구두"
,"열쇠"
,"사물함"
,"마스크"
,"마이크"
,"휴대폰"
,"돈"
,"대걸레"
,"목걸이"
,"노트"
,"횡단보도"
,"펜"
,"연필"
,"리모컨"
,"쓰레기통"
,"샤워기"
,"신발"
,"비누"
,"계단"
,"빨대"
,"가로등"
,"TV"
,"칫솔"
,"치약"
,"신호등"
,"나무"
,"전선줄"
,"청소기"
,"창문"
,"미용실"
,"아파트"
,"미술관"
,"편의점"
,"빵집"
,"사막"
,"소방서"
,"낚시터"
,"은행"
,"민속촌"
,"화장실"
,"보건소"
,"유적지"
,"병원"
,"주택"
,"실험실"
,"세탁소"
,"도서관"
,"마트"
,"산"
,"박물관"
,"바다"
,"공원"
,"약국"
,"사진관"
,"놀이공원"
,"방송국"
,"카페"
,"주민센터"
,"공연장"
,"백화점"
,"꽃집"
,"주유소"
,"놀이터"
,"경찰서"
,"수영장"
,"음식점"
,"강"
,"학교"
,"스키장"
,"썰매장"
,"분식집"
,"우주"
,"경기장"
,"문구점"
,"영화관"
,"시장"
,"우체국"
,"결혼식장"
,"동물원"
,"자전거타기"
,"양치하기"
,"옮기기"
,"젓가락질"
,"청소"
,"요리하기"
,"자르기"
,"설거지"
,"그리기"
,"마시기"
,"밥먹기"
,"운동하기"
,"낚시하기"
,"듣기"
,"인사하기"
,"등산하기"
,"포옹하기"
,"뛰기"
,"눞기"
,"명상하기"
,"줍기"
,"놀이"
,"야구하기"
,"농구하기"
,"축구하기"
,"독서하기"
,"달리기"
,"앉기"
,"노래하기"
,"잠자기"
,"웃기"
,"공부하기"
,"수영하기"
,"달리기"
,"말하기"
,"생각하기"
,"던지기"
,"타자치기"
,"걷기"
,"빨래하기"
,"손씻기"
,"물주기"
,"글쓰기"
,"하품하기"
,"비행기"
,"천사"
,"개미"
,"팔"
,"도끼"
,"붕대"
,"야구방망이"
,"박쥐"
,"욕조"
,"해변"
,"블루베리"
,"수염"
,"침대"
,"하프"
,"혁대"
,"빵"
,"브로콜리"
,"양동이"
,"새"
,"버스"
,"풀숲"
,"책"
,"부메랑"
,"팔찌"
,"뇌"
,"선인장"
,"케이크"
,"계산기"
,"달력"
,"낙타"
,"구름"
,"컴파스"
,"컴퓨터"
,"성"
,"첼로"
,"교회"
,"원"
,"양초"
,"대포"
,"카누"
,"쿠키"
,"소파"
,"다이아몬드"
,"식기세척기"
,"돌고래"
,"도넛"
,"문"
,"용"
,"시계"
,"왕관"
,"귀"
,"팔꿈치"
,"봉투"
,"램프"
,"얼굴"
,"깃털"
,"드릴"
,"크레파스"
,"모자"
,"헤드폰"
,"헬리콥터"
,"발"
,"포크"
,"핫도그"
,"집"
,"아이스크림"
,"재킷"
,"감옥"
,"해머"
,"손"
,"키보드"
,"무릎"
,"사다리"
,"햄버거"
,"나뭇입"
,"다리"
,"전구"
,"벼락"
,"소방차"
,"지도"
,"지갑"
,"싱크대"
,"해골"
,"빌딩"
,"울타리"
,"손가락"
,"잔디"
,"기타"
,"치아"
,"성냥"
,"토네이도"
,"달"
,"모기"
,"기차"
,"나무"
,"삼각형"
,"우산"
,"쥐"
,"입"
,"립스틱"
,"손톱"
,"목걸이"
,"코"
,"테이블"
,"팔각형"
,"칼"
,"티셔츠"
,"오븐"
,"부엉이"
,"붓"
,"베개"
,"피자"
,"연못"
,"낙하산"
,"여권"
,"땅콩"
,"코뿔소"
,"펭귄"
,"피아노"
,"액자"
,"라디오"
,"비"
,"무지개"
,"리모콘"
,"클립"
,"판다"
,"롤러코스터"
,"샌드위치"
,"톱"
,"가위"
,"전갈"
,"드라이버"
,"와인잔"
,"양"
,"얼룩말"
,"별"
,"스테이크"
,"잠수함"
,"풍차"
,"바퀴"
,"눈송이"
,"축구공"
,"눈사람"
,"양말"
,"거미"
,"숟가락"
,"태양"
,"백조"]



def handle_uploaded_file(f):
    # Save file to local directory
    filename = f.name
    fs = FileSystemStorage()
    fs.save(filename, f)
    # Return the file path
    return fs.path(filename)




# Create your views here.
@api_view(['POST'])
def audio_post(request):
    serializer = AudioSerializer(data=request.data)
    if serializer.is_valid():
        # handle the uploaded file
        audio_file = serializer.validated_data['audio_file']

        path = handle_uploaded_file(audio_file)
        

        # 예측 파트
        feature = parse_audio('/home/ubuntu/aimodule/djago_ai/django_ai/media/record.wav', del_silence=True)
        input_length = torch.LongTensor([len(feature)])
        vocab = KsponSpeechVocabulary('/home/ubuntu/aimodule/djago_ai/django_ai/sttmodule/label/ksponspeech_character_vocabs.csv')

        model = torch.load('/home/ubuntu/aimodule/djago_ai/django_ai/sttmodule/model/model.pt', map_location=lambda storage, loc: storage).to('cpu')
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()

        if isinstance(model, ListenAttendSpell):
            model.encoder.device = 'cpu'
            model.decoder.device = 'cpu'

            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, DeepSpeech2):
            model.device = 'cpu'
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
            y_hats = model.recognize(feature.unsqueeze(0), input_length)

        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        # # ...
        dummy_word_lst = ['사과', '너구리', '수박', '쥐', '별', '번개', '달팽이' ]
        some_rand_int = random.randint(0,6)
        result_word = dummy_word_lst[some_rand_int]

        min_distance = 50
        tmp_string = ''
        sentence[0] = sentence[0].replace('!','')
        sentence[0] = sentence[0].replace('?','')
        sentence[0] = sentence[0].replace('@','')
        sentence[0] = sentence[0].replace('#','')
        sentence[0] = sentence[0].replace('$','')
        sentence[0] = sentence[0].replace('/','')
        sentence[0] = sentence[0].replace('%','')
        sentence[0] = sentence[0].replace('.','')
        sentence[0] = sentence[0].replace(' ','')
        print(sentence[0])
        for word in word_lst:
            tmp_distance = jamo_levenshtein(sentence[0], word)
            if tmp_distance < min_distance:
                min_distance = tmp_distance
                tmp_string = word

        print(min_distance, tmp_string)

        fs = FileSystemStorage()
        fs.delete(path)
        
        return Response(data = tmp_string)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)