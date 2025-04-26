package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_1_Test {

    @Test
    public void testToString() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRating("Excellent");
        feedback.setFeedbackComments("This product is very good.");
        feedback.setFeedbackDate("12/12/2012");
        feedback.setFeedbackRater("John Doe");
        String expected = "--------------- \n" + "Rater = John Doe \n" + "Rating = Excellent \n" + "Comments = This product is very good. \n" + "Date = 12/12/2012 \n" + "--------------- \n";
        String actual = feedback.toString();
        assertEquals(expected, actual);
    }
}
