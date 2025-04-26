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
    public void testToString() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Great feedback!");
        feedback.setFeedbackDate("2022-01-01");
        feedback.setFeedbackRater("John Doe");
        String expected = "--------------- \n" + "Rater = John Doe\n" + "Rating = 5\n" + "Comments = Great feedback!\n" + "Date = 2022-01-01\n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }
}
