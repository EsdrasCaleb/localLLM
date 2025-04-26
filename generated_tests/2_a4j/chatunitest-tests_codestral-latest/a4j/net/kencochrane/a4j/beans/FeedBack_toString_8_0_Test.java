package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    @InjectMocks
    private FeedBack feedBack;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        // Arrange
        feedBack.setFeedbackRater("John Doe");
        feedBack.setFeedbackRating("5");
        feedBack.setFeedbackComments("Great job!");
        feedBack.setFeedbackDate("2023-10-01");
        String expected = "--------------- \n" + "Rater = John Doe\n" + "Rating = 5\n" + "Comments = Great job!\n" + "Date = 2023-10-01\n" + "--------------- \n";
        // Act
        String result = feedBack.toString();
        // Assert
        assertEquals(expected, result);
    }
}
