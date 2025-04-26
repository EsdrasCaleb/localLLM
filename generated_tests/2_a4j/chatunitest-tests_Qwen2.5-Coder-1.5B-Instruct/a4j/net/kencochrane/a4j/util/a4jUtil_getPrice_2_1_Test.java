package net.kencochrane.a4j.util;

import java.math.BigDecimal;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

class a4jUtil_getPrice_2_1_Test {

    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        a4jUtil = new a4jUtil();
    }

    @Test
    public void testGetPriceWithValidInput() throws Exception {
        String price = "123.45";
        BigDecimal expected = new BigDecimal("123.45");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 123.45");
    }

    @Test
    public void testGetPriceWithInvalidInput() throws Exception {
        String price = "abc.12";
        BigDecimal expected = new BigDecimal("0.00");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 0.00");
    }

    @Test
    public void testGetPriceWithDecimalPointOnly() throws Exception {
        String price = "1.";
        BigDecimal expected = new BigDecimal("1.00");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 1.00");
    }

    @Test
    public void testGetPriceWithLeadingZeroes() throws Exception {
        String price = "001.23";
        BigDecimal expected = new BigDecimal("1.23");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 1.23");
    }

    @Test
    public void testGetPriceWithTrailingZeros() throws Exception {
        String price = "123.00";
        BigDecimal expected = new BigDecimal("123.00");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 123.00");
    }

    @Test
    public void testGetPriceWithNegativeNumber() throws Exception {
        String price = "-123.45";
        BigDecimal expected = new BigDecimal("-123.45");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be -123.45");
    }

    @Test
    public void testGetPriceWithScientificNotation() throws Exception {
        String price = "1.23e-2";
        BigDecimal expected = new BigDecimal("0.0123");
        assertEquals(expected, a4jUtil.getPrice(price), "Expected price to be 0.0123");
    }
}
