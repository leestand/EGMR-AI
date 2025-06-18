# EGMR-AI(이거먹을래): 
# Everyone's Group Meal Recommendation Chatbot

이 프로젝트는 다중 사용자의 취향을 고려하여 음식을 추천하는 AI 기반 챗봇 서비스입니다.   
사용자의 취향, 메뉴, 위치 데이터를 활용해 LLM 기반으로 식당을 추천하고, 프론트엔드 UI를 통해 대화형 서비스를 제공합니다.

## 📑 발표자료

👉 [EGMR-AI 서비스 발표자료 보러가기](./LeeseoAn_Portfolio_EGMRAI.pdf)


## 🔧 프로젝트 구조
```bash
project-root/
├── README.md
├── requirements.txt          # Python 의존성
├── package.json              # (UI용) Node.js 의존성
├── package-lock.json
├── AI생성코드/
│   └── llm-chatbot/
│       ├── main.py
│       ├── config.py
│       ├── outline.txt
│       ├── api/
│       ├── data/
│       ├── models/
│       └── utils/
├── UI코드/
│   ├── index.js
│   ├── node_modules/
│   ├── package.json
│   ├── package-lock.json
│   └── egmr_ai/
├── 데이터수집코드/
│   ├── NaverReviewSentimentAnalysis.ipynb
│   ├── NaverWebCrawling_address_specific.ipynb
│   ├── NaverWebCrawling_category_menu.ipynb
│   └── NaverWebCrawling_metadata.ipynb
```

## 🧠 주요 기술 스택

- LLM: OpenAI GPT + KoSBERT 기반 임베딩
- 프레임워크: LangChain, FastAPI, Streamlit
- 데이터 수집: Selenium, BeautifulSoup, Pandas
- UI: React.js, Node.js
- 기타: Jupyter Notebook, VSCode

## 🚀 실행 방법

### 1. 📄 AI 서버 실행

```bash
cd AI생성코드/llm-chatbot
pip install -r requirements.txt
python main.py
```

### 2. 📄 UI 서버 실행
```bash
cd UI코드
npm install
npm run dev
```

### 3. 📄 gitignore 설정

아래 파일 및 폴더는 Git에 포함하지 않도록 `.gitignore` 파일에 추가되어 있습니다:
```bash
node_modules/
.ipynb_checkpoints/
pycache/
.env
.DS_Store
```


