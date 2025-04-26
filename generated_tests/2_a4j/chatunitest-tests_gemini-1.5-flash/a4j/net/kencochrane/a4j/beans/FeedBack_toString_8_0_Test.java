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
    void testToString_allFieldsSet() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John Doe");
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Excellent service!");
        feedback.setFeedbackDate("2024-07-26");
        String expectedOutput = "--------------- \n" + "Rater = John Doe\n" + "Rating = 5\n" + "Comments = Excellent service!\n" + "Date = 2024-07-26\n" + "--------------- \n";
        assertEquals(expectedOutput, feedback.toString());
    }

    @Test
    void testToString_emptyFields() {
        FeedBack feedback = new FeedBack();
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = null\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        assertEquals(expectedOutput, feedback.toString());
    }

    @Test
    void testToString_someFieldsSet() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("Jane Doe");
        feedback.setFeedbackComments("Good service.");
        String expectedOutput = "--------------- \n" + "Rater = Jane Doe\n" + "Rating = null\n" + "Comments = Good service.\n" + "Date = null\n" + "--------------- \n";
        assertEquals(expectedOutput, feedback.toString());
    }

    @Test
    void testToString_nullFields() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater(null);
        feedback.setFeedbackRating(null);
        feedback.setFeedbackComments(null);
        feedback.setFeedbackDate(null);
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = null\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        assertEquals(expectedOutput, feedback.toString());
    }

    @Test
    void testToString_fieldsWithSpaces() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John  Doe");
        feedback.setFeedbackRating(" 5 ");
        feedback.setFeedbackComments(" Excellent service! ");
        feedback.setFeedbackDate(" 2024-07-26 ");
        String expectedOutput = "--------------- \n" + "Rater = John  Doe\n" + "Rating =  5 \n" + "Comments =  Excellent service! \n" + "Date =  2024-07-26 \n" + "--------------- \n";
        assertEquals(expectedOutput, feedback.toString());
    }
}
