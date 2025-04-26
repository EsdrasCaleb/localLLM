package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    @Test
    public void testToString_allFieldsPresent() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRater("John Doe");
        feedback.setFeedbackRating("Excellent");
        feedback.setFeedbackComments("Great job!");
        feedback.setFeedbackDate("2024-07-26");
        String expectedOutput = "--------------- \n" + "Rater = John Doe\n" + "Rating = Excellent\n" + "Comments = Great job!\n" + "Date = 2024-07-26\n" + "--------------- \n";
        String actualOutput = feedback.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_emptyFields() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        FeedBack feedback = new FeedBack();
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = null\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        String actualOutput = feedback.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_nullRater() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        FeedBack feedback = new FeedBack();
        feedback.setFeedbackRating("Good");
        feedback.setFeedbackComments("Some comments");
        feedback.setFeedbackDate("2024-07-25");
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = Good\n" + "Comments = Some comments\n" + "Date = 2024-07-25\n" + "--------------- \n";
        String actualOutput = feedback.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }
}
