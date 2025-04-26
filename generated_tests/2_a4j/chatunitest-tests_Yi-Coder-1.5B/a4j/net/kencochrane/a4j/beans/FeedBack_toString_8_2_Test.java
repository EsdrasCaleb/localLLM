package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_2_Test {

    FeedBack fb;

    @BeforeEach
    public void setup() {
        fb = new FeedBack();
    }

    @Test
    public void testToString() {
        fb.setFeedbackRating("5");
        fb.setFeedbackComments("Great work!");
        fb.setFeedbackDate("2023-06-07");
        fb.setFeedbackRater("John Doe");
        String expected = "--------------- \n" + "Rater = John Doe \n" + "Rating = 5 \n" + "Comments = Great work! \n" + "Date = 2023-06-07 \n" + "--------------- \n";
        String actual = fb.toString();
        assertEquals(expected, actual);
    }
}
