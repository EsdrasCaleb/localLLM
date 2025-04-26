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
    public void testToString() {
        // Create an instance of the FeedBack class
        FeedBack feedback = new FeedBack();
        // Set the values for the fields
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Great job!");
        feedback.setFeedbackDate("2023-04-01");
        feedback.setFeedbackRater("John Doe");
        // Call the toString method
        String output = feedback.toString();
        // Assert the output matches the expected string
        Assertions.assertEquals("--------------- \n", output);
        Assertions.assertEquals("Rater = John Doe\n", feedback.getFeedbackRater());
        Assertions.assertEquals("Rating = 5\n", feedback.getFeedbackRating());
        Assertions.assertEquals("Comments = Great job!\n", feedback.getFeedbackComments());
        Assertions.assertEquals("Date = 2023-04-01\n", feedback.getFeedbackDate());
        Assertions.assertEquals("--------------- \n", output);
    }
}
