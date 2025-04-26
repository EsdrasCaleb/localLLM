package net.kencochrane.a4j.beans;

import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class CustomerReview_toString_6_0_Test {

    private CustomerReview customerReview;

    @Test
    public void testToString() {
        customerReview = new CustomerReview();
        customerReview.setRating("A");
        customerReview.setSummary("Good");
        customerReview.setComment("Excellent");
        String expected = "A\nGood\nExcellent\n";
        String actual = customerReview.toString();
        assertEquals(expected, actual);
    }
}
