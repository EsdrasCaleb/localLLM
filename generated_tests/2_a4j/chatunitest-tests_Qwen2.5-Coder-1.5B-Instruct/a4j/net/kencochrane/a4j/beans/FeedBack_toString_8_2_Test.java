package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_2_Test {

    @Test
    public void testToString() throws NoSuchFieldException, IllegalAccessException {
        // Create an instance of the class to be tested
        FeedBack feedBack = new FeedBack();
        // Set up the expected values for the feedback properties
        feedBack.setFeedbackRating("4");
        feedBack.setFeedbackComments("Great work!");
        feedBack.setFeedbackDate("2023-10-05");
        feedBack.setFeedbackRater("John Doe");
        // Call the toString method on the mock object
        String result = feedBack.toString();
        // Verify the correctness of the result
        assertEquals("--------------- \n" + "Rater = John Doe\n" + "Rating = 4\n" + "Comments = Great work!\n" + "Date = 2023-10-05\n" + "--------------- \n", result);
    }
}
