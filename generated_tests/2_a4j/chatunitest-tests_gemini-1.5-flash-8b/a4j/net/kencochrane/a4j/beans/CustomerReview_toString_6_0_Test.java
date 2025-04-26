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

class CustomerReview_toString_6_0_Test {

    private CustomerReview customerReview;

    @BeforeEach
    void setUp() {
        customerReview = new CustomerReview();
    }

    @Test
    void testToString_allFieldsNull() {
        assertEquals("", customerReview.toString());
    }

    @Test
    void testToString_ratingNotNull() {
        customerReview.setRating("5");
        assertEquals("5\n\n", customerReview.toString());
    }

    @Test
    void testToString_summaryNotNull() {
        customerReview.setSummary("Good product");
        assertEquals("\nGood product\n", customerReview.toString());
    }

    @Test
    void testToString_commentNotNull() {
        customerReview.setComment("Excellent!");
        assertEquals("\n\nExcellent!\n", customerReview.toString());
    }

    @Test
    void testToString_allFieldsNotNull() {
        customerReview.setRating("4");
        customerReview.setSummary("Average");
        customerReview.setComment("OK");
        assertEquals("4\nAverage\nOK\n", customerReview.toString());
    }

    @Test
    void testToString_commentWithHtmlTags() {
        customerReview.setComment("<p>This is a test comment.</p><div>more text</div>");
        assertEquals("\n\nThis is a test comment.more text\n", customerReview.toString());
    }
}
