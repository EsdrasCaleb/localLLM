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
    void testToString_withFeedbacks() throws NoSuchFieldException, IllegalAccessException {
        FeedBack[] feedbacks = { new FeedBack("Great product!", 5), new FeedBack("Could be better", 3) };
        Field feedbacksField = SellerFeedback.class.getDeclaredField("feedbacks");
        feedbacksField.setAccessible(true);
        feedbacksField.set(sellerFeedback, new ArrayList<>(Arrays.asList(feedbacks)));
        String expectedOutput = "Comment: Great product!, Rating: 5\n" + "Comment: Could be better, Rating: 3\n" + "# of feedbacks = 2";
        assertEquals(expectedOutput, sellerFeedback.toString());
    }

    @Test
    void testToString_withNoFeedbacks() throws NoSuchFieldException, IllegalAccessException {
        Field feedbacksField = SellerFeedback.class.getDeclaredField("feedbacks");
        feedbacksField.setAccessible(true);
        feedbacksField.set(sellerFeedback, new ArrayList<>());
        assertEquals("feedbacks is null ", sellerFeedback.toString());
    }

    @Test
    void testToString_withNullFeedbacks() throws NoSuchFieldException, IllegalAccessException {
        Field feedbacksField = SellerFeedback.class.getDeclaredField("feedbacks");
        feedbacksField.setAccessible(true);
        feedbacksField.set(sellerFeedback, null);
        assertEquals("feedbacks is null ", sellerFeedback.toString());
    }

    @Test
    void testToString_withOneFeedback() throws NoSuchFieldException, IllegalAccessException {
        FeedBack[] feedbacks = { new FeedBack("Excellent!", 5) };
        Field feedbacksField = SellerFeedback.class.getDeclaredField("feedbacks");
        feedbacksField.setAccessible(true);
        feedbacksField.set(sellerFeedback, new ArrayList<>(Arrays.asList(feedbacks)));
        String expectedOutput = "Comment: Excellent!, Rating: 5\n" + "# of feedbacks = 1";
        assertEquals(expectedOutput, sellerFeedback.toString());
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
