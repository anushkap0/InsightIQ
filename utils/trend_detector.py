import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

class TrendDetector:
    """Detect emerging trends in customer reviews"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def detect_trends(self, df):
        """Detect trends in sentiment over time"""
        try:
            # Convert date to datetime
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            #  daily sentiment ratio
            daily_sentiment = df.groupby('date')['sentiment'].apply(
                lambda x: (x == 'POSITIVE').sum() / len(x)
            )
            
            trends = []
            
            # Detect increasing/decreasing trends
            if len(daily_sentiment) > 2:
                dates = daily_sentiment.index
                values = daily_sentiment.values
                
                # Simple trend detection
                positive_ratio_start = values[0]
                positive_ratio_end = values[-1]
                
                if positive_ratio_end - positive_ratio_start > 0.1:
                    trends.append({
                        "topic": "Overall Satisfaction",
                        "description": "Customer satisfaction is improving over time",
                        "direction": "increasing",
                        "confidence": abs(positive_ratio_end - positive_ratio_start),
                        "keywords": ["satisfaction", "improving", "better"],
                        "timeline": {
                            "dates": dates.strftime('%Y-%m-%d'),
                            "positive_ratio": values
                        }
                    })
                elif positive_ratio_end - positive_ratio_start < -0.1:
                    trends.append({
                        "topic": "Overall Satisfaction",
                        "description": "Customer satisfaction is declining",
                        "direction": "decreasing",
                        "confidence": abs(positive_ratio_end - positive_ratio_start),
                        "keywords": ["declining", "worse", "problem"],
                        "timeline": {
                            "dates": dates.strftime('%Y-%m-%d'),
                            "positive_ratio": values
                        }
                    })
            
            return trends
            
        except Exception as e:
            print(f"Error detecting trends: {e}")
            return []
    
    def generate_recommendations(self, df):
        """Generate business recommendations based on sentiment"""
        recommendations = []
        
        
        negative_reviews = df[df['sentiment'] == 'NEGATIVE']['review_text']
        
        
        issues = {
            "delivery": "delivery|slow|late|time|waiting",
            "food_quality": "cold|bad taste|overcooked|undercooked|quality",
            "pricing": "overpriced|expensive|price|worth",
            "service": "staff|service|friendly|unfriendly|accommodating"
        }
        
        for category, pattern in issues.items():
            count = negative_reviews.str.contains(pattern, case=False, regex=True).sum()
            if count > 2:
                recommendations.append({
                    "category": category.upper(),
                    "recommendation": f"{count} negative reviews mention {category}. Focus on improving this area."
                })
        
        return recommendations