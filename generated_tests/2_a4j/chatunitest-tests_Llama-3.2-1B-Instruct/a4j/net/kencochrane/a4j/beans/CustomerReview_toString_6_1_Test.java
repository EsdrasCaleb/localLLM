package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class CustomerReview_toString_6_1_Test {

    @Test
    public void testToString() {
        CustomerReview review = new CustomerReview();
        review.setRating("4.5");
        review.setSummary("Good service");
        review.setComment("Excellent experience");
        String expectedOutput = "Rating: 4.5\nSummary: Good service\nComment: Excellent experience";
        String actualOutput = review.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
