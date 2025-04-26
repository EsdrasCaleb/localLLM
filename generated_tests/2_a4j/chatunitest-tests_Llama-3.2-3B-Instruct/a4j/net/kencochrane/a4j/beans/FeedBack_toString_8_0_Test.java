package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    @Test
    public void testToString_RaterNotNull() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John");
        String expected = "--------------- \n" + "Rater = John\n" + "Rating = \n" + "Comments = \n" + "Date = \n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }

    @Test
    public void testToString_RatingNotNull() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRating("5");
        String expected = "--------------- \n" + "Rater = \n" + "Rating = 5\n" + "Comments = \n" + "Date = \n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }

    @Test
    public void testToString_CommentsNotNull() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackComments("Great product!");
        String expected = "--------------- \n" + "Rater = \n" + "Rating = \n" + "Comments = Great product!\n" + "Date = \n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }

    @Test
    public void testToString_DateNotNull() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackDate("2022-01-01");
        String expected = "--------------- \n" + "Rater = \n" + "Rating = \n" + "Comments = \n" + "Date = 2022-01-01\n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }

    @Test
    public void testToString_AllNotNull() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John");
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Great product!");
        feedback.setFeedbackDate("2022-01-01");
        String expected = "--------------- \n" + "Rater = John\n" + "Rating = 5\n" + "Comments = Great product!\n" + "Date = 2022-01-01\n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }
}
