package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FeedBack_toString_8_1_Test {

    @Test
    void testToString() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John");
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("This is a test feedback");
        feedback.setFeedbackDate("2022-01-01");
        String expected = "--------------- \n" + "Rater = John\n" + "Rating = 5\n" + "Comments = This is a test feedback\n" + "Date = 2022-01-01\n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }
}
