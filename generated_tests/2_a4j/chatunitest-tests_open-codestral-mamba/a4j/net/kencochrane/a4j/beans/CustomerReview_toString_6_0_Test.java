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

public class CustomerReview_toString_6_0_Test {

    @InjectMocks
    private CustomerReview customerReview;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        customerReview.setRating("5");
        customerReview.setSummary("Great product");
        customerReview.setComment("This is a very good product, I highly recommend it.");
        String expectedToString = "5\nGreat product\nThis is a very good product, I highly recommend it.\n";
        assertEquals(expectedToString, customerReview.toString());
    }
}
