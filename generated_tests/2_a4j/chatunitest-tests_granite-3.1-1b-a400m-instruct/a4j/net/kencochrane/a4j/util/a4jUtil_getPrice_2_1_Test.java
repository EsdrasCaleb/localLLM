package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_getPrice_2_1_Test {

    @Test
    void testGetPrice() {
        a4jUtil util = new a4jUtil();
        // Test with valid input
        assertEquals(123.456789, util.getPrice("123.456789"), "Invalid price format");
        // Test with zero price
        assertEquals(0.00, util.getPrice("0.00"), "Zero price");
        // Test with negative price
        assertEquals(-123.456789, util.getPrice("-123.456789"), "Negative price");
        // Test with price containing non-numeric characters
        assertEquals(0.00, util.getPrice("123.456789abc"), "Price with non-numeric characters");
        // Test with price containing leading/trailing spaces
        assertEquals(123.456789, util.getPrice("  123.456789  "), "Price with leading/trailing spaces");
        // Test with price containing decimal point
        assertEquals(123.456789, util.getPrice("123.456789"), "Price with decimal point");
        // Test with price containing thousands separator
        assertEquals(1234567.89, util.getPrice("1,234,567.89"), "Price with thousands separator");
    }
}
