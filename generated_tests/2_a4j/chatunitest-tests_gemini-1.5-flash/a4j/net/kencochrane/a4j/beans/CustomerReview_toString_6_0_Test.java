package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
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

    @Test
    void testToString_allFieldsSet() {
        CustomerReview review = new CustomerReview();
        review.setRating("5");
        review.setSummary("Excellent product!");
        review.setComment("I highly recommend this product. It's fantastic!");
        String expected = "5\nExcellent product!\nI highly recommend this product. It's fantastic!\n";
        assertEquals(expected, review.toString());
    }

    @Test
    void testToString_nullComment() {
        CustomerReview review = new CustomerReview();
        review.setRating("4");
        review.setSummary("Good product");
        review.setComment(null);
        String expected = "4\nGood product\n\n";
        assertEquals(expected, review.toString());
    }

    @Test
    void testToString_emptyFields() {
        CustomerReview review = new CustomerReview();
        review.setRating("");
        review.setSummary("");
        review.setComment("");
        String expected = "\n\n\n";
        assertEquals(expected, review.toString());
    }

    @Test
    void testToString_commentWithHtmlTags() {
        CustomerReview review = new CustomerReview();
        review.setRating("3");
        review.setSummary("Average");
        review.setComment("This is <a href=\"#\">a link</a> in the comment.");
        String expected = "3\nAverage\nThis is a link in the comment.\n";
        assertEquals(expected, review.toString());
    }

    @Test
    void testToString_commentWithNewLines() {
        CustomerReview review = new CustomerReview();
        review.setRating("2");
        review.setSummary("Poor");
        review.setComment("This is a\nmultiline\ncomment.");
        String expected = "2\nPoor\nThis is amultilinecomment.\n";
        assertEquals(expected, review.toString());
    }
}
