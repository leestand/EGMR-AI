from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Google Gemini 모델 초기화
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0)

def generate_expanded_queries(query, user_preferences, data):
    """
    사용자의 쿼리를 기반으로 확장된 음식점 정보와 리뷰 쿼리를 생성합니다.
    """

    # 사용자 선호도를 기반으로 프롬프트에 추가할 텍스트 생성
    preference_texts = []
    for name, preference in user_preferences.items():
        preference_texts.append(f"{name}: {preference}")
    preference_text = "\n".join(preference_texts)
    
    # 음식점 정보에 관한 확장된 쿼리 생성
    store_info_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """사용자가 요청한 쿼리를 바탕으로 음식점 정보에 관련된 확장된 쿼리를 만들어줘.
            우선 사용자가 언급한 음식 또는 음식 카테고리가 ('술집', '회/해물', '찜/탕/전골', '한식', '족발/보쌈', '고기', '일식', '양식', '중식', '분식', '패스트푸드', '샐러드/샌드위치') 중에서 밀접한 5개를 출력해줘.
            그리고 음식점 이름, 주소, 메뉴, 카테고리와 관련된 정보를 구체적으로 활용해줘.
            """
        ),
        ("human", "{input}")
    ])

    # 리뷰에 관한 확장된 쿼리 생성
    review_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""사용자가 요청한 쿼리를 바탕으로 리뷰와 관련된 확장된 쿼리를 만들어줘.
            음식점의 맛, 서비스, 위생 상태 등의 정보를 바탕으로, 구체적인 리뷰를 반영할 수 있는 확장된 쿼리를 생성해줘.
            그리고 아래 사람들의 선호도를 반영한 쿼리를 생성해줘:

            {preference_text}
            """
        ),
        ("human", "{input}")
    ])

    # LLM을 통해 확장된 음식점 정보 및 리뷰 쿼리 생성
    expanded_store_info_query = (store_info_prompt | llm).invoke({"input": query}).content
    expanded_review_query = (review_prompt | llm).invoke({"input": query}).content

    return expanded_store_info_query, expanded_review_query
