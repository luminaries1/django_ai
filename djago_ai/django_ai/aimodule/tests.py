# -*- coding: utf-8 -*-
from django.test import TestCase


tmp_lst =    [   "사과",
        "아보카도",
        "바나나",
        "피망",
        "브로콜리",
        "양배추",
        "당근",
        "자두",
        "옥수수",
        "오이",
        "가지",
        "마늘",
        "생강",
        "포도",
        "청포도",
        "대파",
        "대추",
        "키위",
        "레몬",
        "상추",
        "귤",
        "망고",
        "멜론",
        "버섯",
        "양배추",
        "양파",
        "오렌지",
        "참외",
        "파프리카",
        "완두콩",
        "복숭아",
        "배",
        "고추",
        "감",
        "파인애플",
        "감자",
        "호박",
        "무",
        "시금치",
        "딸기",
        "고구마",
        "토마토",
        "수박",
    "배우",
    "건축가",
    "군인",
    "화가",
    "우주비행사",
    "작가",
    "농구선수",
    "경호원",
    "촬영기사",
    "카레이서",
    "목수",
    "치어리더",
    "요리사",
    "환경미화원",
    "작곡가",
    "지휘자",
    "상담사",
    "댄서",
    "집배원",
    "개발자",
    "영화감독",
    "의사",
    "농부",
    "소방관",
    "어부",
    "미용사",
    "헬스트레이너",
    "야구선수",
    "기자",
    "판사",
    "마술사",
    "마사지사",
    "정비사",
    "광부",
    "스님",
    "수녀",
    "간호사",
    "피아니스트",
    "조종사",
    "경찰",
    "프로게이머",
    "과학자",
    "판매원",
    "가수",
    "축구선수",
    "승무원",
    "학생",
    "선생님",
    "웨이터",
    "유튜버",
    "악어",
    "곰",
    "벌",
    "나비",
    "카멜레온",
    "고양이",
    "닭",
    "치타",
    "소",
    "게",
    "달팽이",
    "사슴",
    "두더지",
    "개",
    "오리",
    "코끼리",
    "물고기",
    "여우",
    "개구리",
    "기린",
    "해파리",
    "하마",
    "하이에나",
    "말",
    "잠자리",
    "캥거루",
    "사자",
    "병아리",
    "미어캣",
    "메뚜기",
    "원숭이",
    "쥐",
    "무당벌레",
    "너구리",
    "문어",
    "오랑우탄",
    "올빼미",
    "팬더",
    "돼지",
    "풍뎅이",
    "토끼",
    "사마귀",
    "상어",
    "양",
    "뱀",
    "호랑이",
    "거북이",
    "고래",
    "늑대",
    "염소",
    "에어컨",
    "자동차",
    "백팩",
    "바구니",
    "자전거",
    "칠판",
    "카메라",
    "카드",
    "의자",
    "바지",
    "노트북",
    "컵",
    "커튼",
    "책상",
    "문",
    "이어폰",
    "엘리베이터",
    "지우개",
    "비상문",
    "안경",
    "꽃",
    "냉장고",
    "구두",
    "열쇠",
    "사물함",
    "마스크",
    "마이크",
    "휴대폰",
    "돈",
    "대걸레",
    "목걸이",
    "노트",
    "횡단보도",
    "펜",
    "연필",
    "리모컨",
    "쓰레기통",
    "샤워기",
    "신발",
    "비누",
    "계단",
    "빨대",
    "가로등",
    "TV",
    "칫솔",
    "치약",
    "신호등",
    "나무",
    "전선줄",
    "청소기",
    "창문",
    "미용실",
    "아파트",
    "미술관",
    "편의점",
    "빵집",
    "사막",
    "소방서",
    "낚시터",
    "은행",
    "민속촌",
    "화장실",
    "보건소",
    "유적지",
    "병원",
    "주택",
    "실험실",
    "세탁소",
    "도서관",
    "마트",
    "산",
    "박물관",
    "바다",
    "공원",
    "약국",
    "사진관",
    "놀이공원",
    "방송국",
    "카페",
    "주민센터",
    "공연장",
    "백화점",
    "꽃집",
    "주유소",
    "놀이터",
    "경찰서",
    "수영장",
    "음식점",
    "강",
    "학교",
    "스키장",
    "썰매장",
    "분식집",
    "우주",
    "경기장",
    "문구점",
    "영화관",
    "시장",
    "우체국",
    "결혼식장",
    "동물원",
    "자전거타기",
    "양치하기",
    "옮기기",
    "젓가락질",
    "청소",
    "요리하기",
    "자르기",
    "설거지",
    "그리기",
    "마시기",
    "밥먹기",
    "운동하기",
    "낚시하기",
    "듣기",
    "인사하기",
    "등산하기",
    "포옹하기",
    "뛰기",
    "눞기",
    "명상하기",
    "줍기",
    "놀이",
    "야구하기",
    "농구하기",
    "축구하기",
    "독서하기",
    "달리기",
    "앉기",
    "노래하기",
    "잠자기",
    "웃기",
    "공부하기",
    "수영하기",
    "달리기",
    "말하기",
    "생각하기",
    "던지기",
    "타자치기",
    "걷기",
    "빨래하기",
    "손씻기",
    "물주기",
    "글쓰기",
    "하품하기"
]

tmp_lst_2 = ["계단","딸기","가로등","호랑이","칫솔","치약","신호등","수박","고래","빗자루","돼지","파인애플","감자","토끼","너구리","강","상어","신발","달팽이","뱀","문어","곰","사과","백팩","바나나","벌","자전거","병원","열쇠","원숭이","양파","나비","카메라","자동차","당근","고양이","의자","캥거루","노트북","산","바지","소","게","악어","컵","오리","코끼리","말","사자","버섯","연필","지우개","안경","물고기","꽃","개구리","기린","포도","마이크","바다"]


for word in tmp_lst_2:
    if word not in tmp_lst:
        print(word)



# Create your tests here.
