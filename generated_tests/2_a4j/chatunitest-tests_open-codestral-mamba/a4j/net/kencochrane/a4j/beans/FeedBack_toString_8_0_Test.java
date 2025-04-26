package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    private FeedBack feedback;

    @BeforeEach
    public void setUp() {
        feedback = new FeedBack();
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Excellent service");
        feedback.setFeedbackDate("2021-10-15");
        feedback.setFeedbackRater("John Doe");
    }

    @Test
    public void testToString() {
        String expected = "--------------- \n" + "Rater = John Doe \n" + "Rating = 5 \n" + "Comments = Excellent service \n" + "Date = 2021-10-15 \n" + "--------------- \n";
        assertEquals(expected, feedback.toString());
    }
}
