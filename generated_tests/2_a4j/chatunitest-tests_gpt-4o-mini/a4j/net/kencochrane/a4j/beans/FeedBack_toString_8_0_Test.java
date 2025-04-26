package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    private FeedBack feedBack;

    @BeforeEach
    public void setUp() {
        feedBack = new FeedBack();
    }

    @Test
    public void testToStringAllFieldsSet() {
        feedBack.setFeedbackRater("John Doe");
        feedBack.setFeedbackRating("5 stars");
        feedBack.setFeedbackComments("Excellent service!");
        feedBack.setFeedbackDate("2023-10-01");
        String expectedOutput = "--------------- \n" + "Rater = John Doe\n" + "Rating = 5 stars\n" + "Comments = Excellent service!\n" + "Date = 2023-10-01\n" + "--------------- \n";
        assertEquals(expectedOutput, feedBack.toString());
    }

    @Test
    public void testToStringWithNullFields() {
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = null\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        assertEquals(expectedOutput, feedBack.toString());
    }

    @Test
    public void testToStringWithEmptyFields() {
        feedBack.setFeedbackRater("");
        feedBack.setFeedbackRating("");
        feedBack.setFeedbackComments("");
        feedBack.setFeedbackDate("");
        String expectedOutput = "--------------- \n" + "Rater = \n" + "Rating = \n" + "Comments = \n" + "Date = \n" + "--------------- \n";
        assertEquals(expectedOutput, feedBack.toString());
    }
}
