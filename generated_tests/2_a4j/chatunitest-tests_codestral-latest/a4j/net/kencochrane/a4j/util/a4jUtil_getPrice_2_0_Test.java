package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
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

class a4jUtil_getPrice_2_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetPriceValidInput() {
        String price = "123.45";
        BigDecimal expected = new BigDecimal("123.45");
        BigDecimal result = a4jUtil.getPrice(price);
        assertEquals(expected, result);
    }

    @Test
    void testGetPriceWithSpaces() {
        String price = "  123.45  ";
        BigDecimal expected = new BigDecimal("123.45");
        BigDecimal result = a4jUtil.getPrice(price);
        assertEquals(expected, result);
    }

    @Test
    void testGetPriceWithInvalidCharacters() {
        String price = "123.45abc";
        BigDecimal expected = new BigDecimal("123.45");
        BigDecimal result = a4jUtil.getPrice(price);
        assertEquals(expected, result);
    }

    @Test
    void testStripString() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("stripString", String.class, String.class);
        method.setAccessible(true);
        String allowedString = ".0123456789";
        String input = "123.45abc";
        String expected = "123.45";
        String result = (String) method.invoke(a4jUtil, allowedString, input);
        assertEquals(expected, result);
    }
}
