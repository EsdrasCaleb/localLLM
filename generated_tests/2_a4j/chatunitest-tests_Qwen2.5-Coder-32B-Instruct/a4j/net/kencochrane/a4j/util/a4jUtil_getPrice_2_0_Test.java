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

public class a4jUtil_getPrice_2_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetPrice_ValidInput() throws Exception {
        // Arrange
        String input = "123.45abc";
        BigDecimal expectedOutput = new BigDecimal("123.45");
        // Act
        BigDecimal actualOutput = a4jUtil.getPrice(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testGetPrice_ZeroPrice() throws Exception {
        // Arrange
        String input = "0.00";
        BigDecimal expectedOutput = new BigDecimal("0.00");
        // Act
        BigDecimal actualOutput = a4jUtil.getPrice(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testGetPrice_LargePrice() throws Exception {
        // Arrange
        String input = "1234567890.12";
        BigDecimal expectedOutput = new BigDecimal("1234567890.12");
        // Act
        BigDecimal actualOutput = a4jUtil.getPrice(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testStripString_PrivateMethod() throws Exception {
        // Arrange
        String allowedString = ".0123456789";
        String input = "123.45abc";
        String expectedOutput = "123.45";
        // Access private method using reflection
        Method method = a4jUtil.getClass().getDeclaredMethod("stripString", String.class, String.class);
        method.setAccessible(true);
        // Act
        String actualOutput = (String) method.invoke(a4jUtil, allowedString, input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
