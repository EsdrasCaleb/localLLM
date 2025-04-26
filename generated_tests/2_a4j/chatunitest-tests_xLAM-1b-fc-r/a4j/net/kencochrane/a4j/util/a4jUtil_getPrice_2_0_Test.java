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

public class a4jUtil_getPrice_2_0_Test {

    @Test
    public void testGetPrice() {
        // Arrange
        a4jUtil util = new a4jUtil();
        String price = "123.45";
        // Act
        BigDecimal result = util.getPrice(price);
        // Assert
        assertEquals(new BigDecimal("123.45"), result);
    }

    @Test
    public void testGetPriceWithNonNumeric() {
        // Arrange
        a4jUtil util = new a4jUtil();
        String price = "123.45abc";
        // Act
        BigDecimal result = util.getPrice(price);
        // Assert
        assertEquals(new BigDecimal("123.45"), result);
    }

    @Test
    public void testGetPriceWithZero() {
        // Arrange
        a4jUtil util = new a4jUtil();
        String price = "0";
        // Act
        BigDecimal result = util.getPrice(price);
        // Assert
        assertEquals(new BigDecimal("0.00"), result);
    }
}
