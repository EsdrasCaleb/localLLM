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

    private CustomerReview customerReview;

    private a4jUtil mockA4jUtil;

    @BeforeEach
    public void setUp() {
        customerReview = new CustomerReview();
        mockA4jUtil = Mockito.mock(a4jUtil.class);
        customerReview.jawsUtil = mockA4jUtil;
    }

    @Test
    public void testToString_AllFieldsSet() throws Exception {
        // Set values for all fields
        setField(customerReview, "rating", "5");
        setField(customerReview, "summary", "Great Product");
        setField(customerReview, "comment", "This product is <b>awesome</b>!");
        // Call the method under test
        String result = customerReview.toString();
        // Verify the result
        assertEquals("5\nGreat Product\nThis product is awesome!\n", result);
    }

    @Test
    public void testToString_SomeFieldsNull() throws Exception {
        // Set values for some fields and leave others null
        setField(customerReview, "rating", null);
        setField(customerReview, "summary", "Average");
        setField(customerReview, "comment", null);
        // Call the method under test
        String result = customerReview.toString();
        // Verify the result
        assertEquals("null\nAverage\nnull\n", result);
    }

    @Test
    public void testToString_AllFieldsNull() throws Exception {
        // Leave all fields null
        setField(customerReview, "rating", null);
        setField(customerReview, "summary", null);
        setField(customerReview, "comment", null);
        // Call the method under test
        String result = customerReview.toString();
        // Verify the result
        assertEquals("null\nnull\nnull\n", result);
    }

    @Test
    public void testToString_CommentWithHTMLTags() throws Exception {
        // Set values for all fields, including a comment with HTML tags
        setField(customerReview, "rating", "3");
        setField(customerReview, "summary", "Okay");
        setField(customerReview, "comment", "This <strong>product</strong> is <em>okay</em>.");
        // Call the method under test
        String result = customerReview.toString();
        // Verify the result
        assertEquals("3\nOkay\nThis product is okay.\n", result);
    }

    private void setField(Object obj, String fieldName, Object value) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }
}
