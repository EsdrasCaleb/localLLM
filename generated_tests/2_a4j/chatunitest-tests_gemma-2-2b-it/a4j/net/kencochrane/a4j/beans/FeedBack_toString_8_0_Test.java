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
    void testToString() {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John Doe");
        feedback.setFeedbackRating("5");
        feedback.setFeedbackComments("Good work!");
        feedback.setFeedbackDate("2023-10-26");
        String expectedOutput = "--------------- \nRater = John Doe\nRating = 5\nComments = Good work!\nDate = 2023-10-26\n--------------- \n";
        String actualOutput = feedback.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
