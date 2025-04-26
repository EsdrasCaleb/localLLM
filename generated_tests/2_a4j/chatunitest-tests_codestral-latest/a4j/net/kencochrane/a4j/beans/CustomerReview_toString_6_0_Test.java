package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class CustomerReview_toString_6_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private CustomerReview customerReview;

    @BeforeEach
    public void setUp() {
        customerReview.setRating("5");
        customerReview.setSummary("Great product!");
        customerReview.setComment("I love it!");
    }

    @Test
    public void testToString() {
        String expected = "5\nGreat product!\nI love it!\n";
        String actual = customerReview.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithNullValues() {
        customerReview.setRating(null);
        customerReview.setSummary(null);
        customerReview.setComment(null);
        String expected = "null\nnull\nnull\n";
        String actual = customerReview.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithEmptyValues() {
        customerReview.setRating("");
        customerReview.setSummary("");
        customerReview.setComment("");
        String expected = "\n\n\n";
        String actual = customerReview.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithHTMLTagsInComment() {
        customerReview.setComment("<b>I love it!</b>");
        String expected = "5\nGreat product!\nI love it!\n";
        String actual = customerReview.toString();
        assertEquals(expected, actual);
    }
}
