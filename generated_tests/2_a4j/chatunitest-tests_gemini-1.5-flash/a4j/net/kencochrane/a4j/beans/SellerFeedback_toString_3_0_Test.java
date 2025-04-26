package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerFeedback_toString_3_0_Test {

    private SellerFeedback sellerFeedback;

    @BeforeEach
    void setUp() {
        sellerFeedback = new SellerFeedback();
    }

    @Test
    void testToString_withNullFeedbacks() throws NoSuchFieldException, IllegalAccessException {
        Field feedbacksField = SellerFeedback.class.getDeclaredField("feedbacks");
        feedbacksField.setAccessible(true);
        feedbacksField.set(sellerFeedback, null);
        assertEquals("feedbacks is null ", sellerFeedback.toString());
    }
}

class FeedBack {

    private String comment;

    private int rating;

    public FeedBack() {
    }

    public FeedBack(String comment, int rating) {
        this.comment = comment;
        this.rating = rating;
    }

    @Override
    public String toString() {
        return "Comment: " + comment + ", Rating: " + rating;
    }
}

class SellerFeedback {

    private ArrayList<FeedBack> feedbacks;

    public SellerFeedback() {
    }

    @Override
    public String toString() {
        if (feedbacks == null || feedbacks.isEmpty()) {
            return "feedbacks is null ";
        } else {
            StringBuilder sb = new StringBuilder();
            for (FeedBack feedback : feedbacks) {
                sb.append(feedback.toString()).append("\n");
            }
            sb.append("# of feedbacks = ").append(feedbacks.size());
            return sb.toString();
        }
    }
}
